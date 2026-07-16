import os
import argparse
from shutil import copyfile
import torch.distributed as dist
import torch
import datetime
import time
import logging
import numpy as np
import os.path as osp
from collections import OrderedDict

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel

from core.dist_utils import (
    get_world_size,
)
from omegaconf import OmegaConf
from core.train_utils import seed_all, AverageMeterGroups
from core.flow_utils import flow_to_image
from metrics.image_quality_metrics import calculate_psnr
from core.build_utils import build_from_cfg
from core.logger import CustomLogger

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


parser = argparse.ArgumentParser(description='VFI')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('--local_rank', '--local-rank', dest='local_rank',
                    type=int, default=None)

args = parser.parse_args()

def apply_shared_dataset_dir(config):
    """Copy data.dataset_dir into train/val/test params when they omit it."""
    data_cfg = config.get('data')
    if data_cfg is None:
        return config

    dataset_dir = data_cfg.get('dataset_dir')
    if dataset_dir is None:
        return config

    for split in ('train', 'val', 'test'):
        split_cfg = data_cfg.get(split)
        if split_cfg is None:
            continue

        params = split_cfg.get('params')
        if params is None:
            split_cfg['params'] = {}
            params = split_cfg['params']

        if params.get('dataset_dir') is None:
            params['dataset_dir'] = dataset_dir

    return config


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = self.config['local_rank']
        init_log = self._init_logger()
        self._init_dataset()
        self._init_loss()
        self.model_name = config['exp_name']
        self.model = build_from_cfg(config.network).to(self.config.device)

        if config.get('distributed', False):
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)

        init_log += str(self.model)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=config.lr, weight_decay=config.weight_decay)
        if self.rank == 0:
            print(init_log)
        self.logger(init_log)
        self.resume_training()

    def resume_training(self):
        ckpt_path = self.config.get('resume_state')
        if ckpt_path is not None:
            ckpt = torch.load(self.config['resume_state'])
            if isinstance(self.model, (DDP, DataParallel)):
                self.model.module.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optim'])
            self.resume_epoch = ckpt.get('epoch')
            self.logger(
                f'load model from {ckpt_path} and training resumes from epoch {self.resume_epoch}')
        else:
            self.resume_epoch = 0

    def _model_module(self):
        """Return the underlying model module (unwrap DP/DDP if needed)."""
        if isinstance(self.model, (DDP, DataParallel)):
            return self.model.module
        return self.model

    def _maybe_split_branches(self, epoch):
        """Trigger branch split for hybrid models when configured."""
        module = self._model_module()
        maybe_split = getattr(module, 'maybe_split_branches', None)
        if callable(maybe_split):
            split_happened = maybe_split(epoch, self.config.get('epochs'))
            if split_happened and self.rank == 0:
                self.logger(f'Branches split at epoch {epoch+1}.')

    def _init_logger(self):
        init_log = ''
        console_cfg = dict(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"{self.config['save_dir']}/log",
            filemode='w')
        wandb_cfg = None
        use_wandb = self.config['logger'].get('use_wandb', False)
        if use_wandb and wandb is not None:
            resume_id = self.config['logger'].get('resume_id', None)
            project = self.config['logger'].get('project')
            entity = self.config['logger'].get('entity')
            if project:
                if resume_id:
                    wandb_id = resume_id
                    resume = 'allow'
                    init_log += f'Resume wandb logger with id={wandb_id}.'
                else:
                    wandb_id = wandb.util.generate_id()
                    resume = 'never'

                wandb_config = OmegaConf.to_container(self.config, resolve=True)
                wandb_cfg = dict(id=wandb_id,
                                 resume=resume,
                                 name=osp.basename(self.config['save_dir']),
                                 config=wandb_config,
                                 project=project)
                if entity:
                    wandb_cfg['entity'] = entity
                init_log += f'Use wandb logger with id={wandb_id}; project=[{project}]'
                if entity:
                    init_log += f' entity=[{entity}]'
                init_log += '.'
            else:
                init_log += 'W&B disabled: missing project.'
        elif use_wandb and wandb is None:
            init_log += 'W&B disabled: package not installed.'
        self.logger = CustomLogger(
            console_cfg,
            wandb_cfg=wandb_cfg,
            rank=self.rank,
        )
        return init_log

    def _init_dataset(self):
        dataset_train = build_from_cfg(self.config.data.train)
        dataset_val = build_from_cfg(self.config.data.val)
        dataset_test = None

        if self.config.data.get('test') is not None:
            dataset_test = build_from_cfg(self.config.data.test)

        collate_train = getattr(dataset_train, 'collate_fn', None)
        collate_val = getattr(dataset_val, 'collate_fn', None)
        collate_test = getattr(dataset_test, 'collate_fn', None) if dataset_test else None

        train_loader_cfg = dict(self.config.data.train_loader)

        if self.config.get('distributed', False) and self.config.get('world_size', 1) > 1:
            self.sampler = DistributedSampler(
                dataset_train,
                num_replicas=self.config['world_size'],
                rank=self.config['local_rank'],
            )
            train_loader_cfg['batch_size'] //= self.config['world_size']
            train_loader_cfg['sampler'] = self.sampler
            train_loader_cfg.pop('shuffle', None)
        else:
            self.sampler = None
            train_loader_cfg.setdefault('shuffle', True)
            train_loader_cfg.pop('sampler', None)

        self.loader_train = DataLoader(
            dataset_train,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_train,
            **train_loader_cfg,
        )

        self.loader_val = DataLoader(dataset_val, **self.config.data.val_loader,
                                     pin_memory=True, shuffle=False, drop_last=False,
                                     collate_fn=collate_val)
        self.loader_test = None
        if dataset_test is not None:
            test_loader_cfg = dict(self.config.data.get('test_loader', self.config.data.val_loader))
            test_loader_cfg.pop('shuffle', None)
            test_loader_cfg.pop('drop_last', None)
            test_loader_cfg.pop('pin_memory', None)
            self.loader_test = DataLoader(
                dataset_test,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_test,
                **test_loader_cfg,
            )
        if self.rank == 0:
            train_batches = len(self.loader_train)
            train_triplets = len(dataset_train)
            val_batches = len(self.loader_val)
            val_triplets = len(dataset_val)
            self.logger(
                f'Training loader prepared with {train_triplets} triplets across {train_batches} batches.'
            )
            self.logger(
                f'Validation loader prepared with {val_triplets} triplets across {val_batches} batches.'
            )
            if self.loader_test is not None:
                test_batches = len(self.loader_test)
                test_triplets = len(dataset_test)
                self.logger(
                    f'Test loader prepared with {test_triplets} triplets across {test_batches} batches.'
                )

    def _init_loss(self):
        self.loss_dict = dict()
        for loss_cfg in self.config.losses:
            loss = build_from_cfg(loss_cfg)
            self.loss_dict[loss_cfg['nickname']] = loss

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self, iters):
        ratio = 0.5 * (1.0 + np.cos(iters /
                                    (self.config['epochs'] * self.loader_train.__len__()) * np.pi))
        lr = (self.config['lr'] - self.config['lr_min']
              ) * ratio + self.config['lr_min']
        return lr

    def train(self):
        local_rank = self.config['local_rank']
        best_psnr = 0.0
        loss_group = AverageMeterGroups()
        time_group = AverageMeterGroups()
        iters_per_epoch = self.loader_train.__len__()
        iters = self.resume_epoch * iters_per_epoch
        total_iters = self.config['epochs'] * iters_per_epoch
        start_t = time.time()
        total_t = 0
        for epoch in range(self.resume_epoch, self.config['epochs']):
            self._maybe_split_branches(epoch)
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for data in self.loader_train:
                for k, v in data.items():
                    data[k] = v.to(self.config['device'])
                data_t = time.time() - start_t

                lr = self.get_lr(iters)
                self.set_lr(self.optimizer, lr)

                self.optimizer.zero_grad()
                results = self.model(**data)
                total_loss = torch.tensor(0., device=self.config['device'])
                for name, loss in self.loss_dict.items():
                    l = loss(**results, **data)
                    loss_group.update({name: l.cpu().data})
                    total_loss += l
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=1.0)
                self.optimizer.step()

                iters += 1

                iter_t = time.time() - start_t
                total_t += iter_t
                time_group.update({'data_t': data_t, 'iter_t': iter_t})

                # if local_rank == 0:
                #     model_device = next(self.model.parameters()).device
                #     if model_device.type == 'cuda':
                #         torch.cuda.synchronize(model_device)
                #         mem_alloc = torch.cuda.memory_allocated(model_device) / 1e6
                #         mem_reserved = torch.cuda.memory_reserved(model_device) / 1e6
                #         mem_peak = torch.cuda.max_memory_allocated(model_device) / 1e6
                #         mem_peak_reserved = torch.cuda.max_memory_reserved(model_device) / 1e6
                #         self.logger(
                #             f"iter {iters + 1}: total_time={total_t:.3f}s "
                #             f"| mem alloc={mem_alloc:.1f}MB reserved={mem_reserved:.1f}MB "
                #             f"peak alloc={mem_peak:.1f}MB peak reserved={mem_peak_reserved:.1f}MB"
                #         )
                #     else:
                #         self.logger(f"iter {iters + 1}: total_time={total_t:.3f}s | mem=cpu")

                if (iters + 1) % 100 == 0 and local_rank == 0:
                    model_device = next(self.model.parameters()).device
                    device_type = 'gpu' if model_device.type == 'cuda' else 'cpu'
                    self.logger(
                        f'iter {iters + 1}: running on {device_type} ({model_device})'
                    )

                if (iters+1) % 100 == 0 and local_rank == 0:
                    tpi = total_t / (iters - self.resume_epoch * iters_per_epoch)
                    eta = total_iters * tpi
                    remainder = (total_iters - iters) * tpi
                    eta = self.eta_format(eta)

                    remainder = self.eta_format(remainder)
                    log_str = f"[{self.model_name}]epoch:{epoch +1}/{self.config['epochs']} "
                    log_str += f"iter:{iters + 1}/{self.config['epochs'] * iters_per_epoch} "
                    log_str += f"time:{time_group.avg('iter_t'):.3f}({time_group.avg('data_t'):.3f}) "
                    log_str += f"lr:{lr:.3e} eta:{remainder}({eta})\n"
                    wandb_metrics = {}
                    for name in self.loss_dict.keys():
                        avg_l = loss_group.avg(name)
                        log_str += f"{name}:{avg_l:.3e} "
                        wandb_metrics[f'loss/{name}'] = avg_l
                    log_str += f'best:{best_psnr:.2f}dB\n\n'
                    self.logger(
                        log_str,
                        wandb_metrics=wandb_metrics,
                        step=iters,
                    )
                    loss_group.reset()
                    time_group.reset()
                start_t = time.time()

            if (epoch+1) % self.config['eval_interval'] == 0 and local_rank == 0:
                psnr, eval_t = self.evaluate(epoch, global_step=iters)
                total_t += eval_t
                if psnr > best_psnr:
                    best_psnr = psnr
                    self.save('psnr_best.pth', epoch)
                    if self.logger.enable_wandb:
                        wandb.run.summary["best_psnr"] = best_psnr

                if (epoch+1) % 50 == 0:
                    self.save(f'epoch_{epoch+1}.pth', epoch)
                self.save('latest.pth', epoch)

        self.logger.close()

    def evaluate(self, epoch, global_step=None):
        psnr_list = []
        time_stamp = time.time()
        was_training = self.model.training
        self.model.eval()

        log_samples = {}
        max_wandb_samples = self.config['logger'].get('wandb_log_samples', 2)
        sample_count = 0

        for i, data in enumerate(self.loader_val):
            for k, v in data.items():
                data[k] = v.to(self.config['device'])

            with torch.no_grad():
                results = self.model(**data)
                imgt_pred = results['imgt_pred']
                imgt_pred_B = results.get('imgt_pred_B')
                imgt_pred_B_A = results.get('imgt_pred_B_A')
                flow0_pred_list = results.get('flow0_pred')
                flow1_pred_list = results.get('flow1_pred')

                for j in range(data['img0'].shape[0]):
                    psnr = calculate_psnr(
                        imgt_pred[j].detach().unsqueeze(0),
                        data['imgt'][j].unsqueeze(0),
                    )
                    psnr_list.append(float(psnr))

                    if (
                        self.rank == 0
                        and getattr(self.logger, 'enable_wandb', False)
                        and sample_count < max_wandb_samples
                    ):
                        tag = f'sample_{sample_count}'

                        pred_img = imgt_pred[j].detach().cpu().clamp_(0, 1)
                        gt_img = data['imgt'][j].detach().cpu().clamp_(0, 1)

                        pred_img_np = pred_img.permute(1, 2, 0).numpy()
                        gt_img_np = gt_img.permute(1, 2, 0).numpy()

                        log_samples[f'eval/{tag}/pred_frame'] = wandb.Image(
                            pred_img_np, caption=f'Predicted frame (epoch {epoch+1})'
                        )
                        if imgt_pred_B is not None:
                            pred_B_tensor = imgt_pred_B[j]
                            pred_B_np = (
                                pred_B_tensor.detach()
                                .cpu()
                                .clamp_(0, 1)
                                .permute(1, 2, 0)
                                .numpy()
                            )
                            log_samples[f'eval/{tag}/pred_frame_B'] = wandb.Image(
                                pred_B_np,
                                caption=f'Orthogonal predicted frame (epoch {epoch+1})',
                            )
                        if imgt_pred_B_A is not None:
                            pred_B_A_tensor = imgt_pred_B_A[j]
                            pred_B_A_np = (
                                pred_B_A_tensor.detach()
                                .cpu()
                                .clamp_(0, 1)
                                .permute(1, 2, 0)
                                .numpy()
                            )
                            log_samples[f'eval/{tag}/pred_frame_B_A'] = wandb.Image(
                                pred_B_A_np,
                                caption=f'Orthogonal->primitive predicted frame (epoch {epoch+1})',
                            )
                        log_samples[f'eval/{tag}/gt_frame'] = wandb.Image(
                            gt_img_np, caption='Ground-truth frame'
                        )

                        flow_gt = data.get('flow')
                        if flow_gt is not None:
                            flow_gt = flow_gt[j].detach().cpu()
                            if flow_gt.shape[0] >= 2:
                                gt_flow0_np = flow_to_image(
                                    flow_gt[0:2].permute(1, 2, 0).numpy()
                                )
                                log_samples[f'eval/{tag}/gt_flow_t0'] = wandb.Image(
                                    gt_flow0_np, caption='Ground-truth flow t0'
                                )
                            if flow_gt.shape[0] >= 4:
                                gt_flow1_np = flow_to_image(
                                    flow_gt[2:4].permute(1, 2, 0).numpy()
                                )
                                log_samples[f'eval/{tag}/gt_flow_t1'] = wandb.Image(
                                    gt_flow1_np, caption='Ground-truth flow t1'
                                )

                        def _prepare_pred_flow(flow_data, flow_name):
                            if flow_data is None:
                                return
                            if isinstance(flow_data, torch.Tensor):
                                flow_tensor = flow_data
                            elif isinstance(flow_data, (list, tuple)):
                                if len(flow_data) == 0:
                                    return
                                flow_tensor = flow_data[0]
                            else:
                                return
                            if not isinstance(flow_tensor, torch.Tensor):
                                return
                            if flow_tensor.dim() == 5:
                                flow_tensor = flow_tensor[j, 0]
                            elif flow_tensor.dim() == 4:
                                flow_tensor = flow_tensor[j]
                            else:
                                return
                            flow_np = flow_tensor.detach().cpu().permute(1, 2, 0).numpy()
                            flow_img = flow_to_image(flow_np)
                            log_samples[f'eval/{tag}/{flow_name}'] = wandb.Image(
                                flow_img, caption=f'{flow_name.replace("_", " ").title()}'
                            )

                        _prepare_pred_flow(flow0_pred_list, 'pred_flow_t0')
                        _prepare_pred_flow(flow1_pred_list, 'pred_flow_t1')

                        sample_count += 1

        eval_time = time.time() - time_stamp

        mean_psnr = float(np.array(psnr_list).mean())

        if getattr(self.logger, 'enable_wandb', False) and wandb is not None:
            log_samples['eval/psnr'] = mean_psnr
            if global_step is not None:
                log_samples['global_step'] = global_step
            wandb.log(log_samples)

        if was_training:
            self.model.train()

        self.logger('eval epoch:{}/{} time:{:.2f} psnr:{:.3f}'.format(
            epoch+1, self.config["epochs"], eval_time, mean_psnr))
        return mean_psnr, eval_time

    def save(self, name, epoch):
        save_path = '{}/{}/{}'.format(self.config['save_dir'], 'ckpts', name)
        ckpt = OrderedDict(epoch=epoch)
        if isinstance(self.model, (DDP, DataParallel)):
            ckpt['state_dict'] = self.model.module.state_dict()
        else:
            ckpt['state_dict'] = self.model.state_dict()
        ckpt['optim'] = self.optimizer.state_dict()
        torch.save(ckpt, save_path)

    def eta_format(self, eta):
        time_str = ''
        if eta >= 3600:
            hours = int(eta // 3600)
            eta -= hours * 3600
            time_str = f'{hours}'

        if eta >= 60:
            mins = int(eta // 60)
            eta -= mins * 60
            time_str = f'{time_str}:{mins:02}'

        eta = int(eta)
        time_str = f'{time_str}:{eta:02}'
        return time_str


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if torch.cuda.is_available():
        print(f'Rank {rank} is available')
        config['device'] = f"cuda:{rank}"
        if config['distributed']:
            dist.init_process_group(backend='nccl', 
                                    timeout=datetime.timedelta(seconds=5400))
    else:
        config['device'] = 'cpu'

    cfg_name = os.path.basename(args.config).split('.')[0]
    config['exp_name'] = cfg_name + '_' + config['exp_name']
    config['save_dir'] = os.path.join(config['save_dir'], config['exp_name'])

    if (not config['distributed']) or rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(f'{config["save_dir"]}/ckpts', exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    print(f'using GPU {rank} for training')
    trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(args.config)
    cfg = apply_shared_dataset_dir(cfg)
    seed_all(cfg.seed)
    rank = args.local_rank
    if rank is None:
        rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(f'cuda:{rank}'))
    # setting distributed configurations
    if cfg.get('distributed', False):
        cfg['world_size'] = get_world_size()
    else:
        cfg['world_size'] = 1
    cfg['local_rank'] = rank
    if rank == 0:
       print('world_size: ', cfg['world_size'])
    main_worker(rank, cfg)
        
