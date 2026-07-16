import time
import shutil
import logging
import os.path as osp

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def mv_archived_logger(name):
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S_", time.localtime())
    basename = 'archived_' + timestamp + osp.basename(name)
    archived_name = osp.join(osp.dirname(name), basename)
    shutil.move(name, archived_name)


class CustomLogger:
    def __init__(self, common_cfg, wandb_cfg=None, rank=0):
        global global_logger
        self.rank = rank
        self.enable_wandb = False

        if self.rank == 0:
            self.logger = logging.getLogger('VFI')
            self.logger.setLevel(logging.INFO)
            format_str = logging.Formatter(common_cfg['format'])

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(format_str)

            if osp.exists(common_cfg['filename']):
                mv_archived_logger(common_cfg['filename'])

            file_handler = logging.FileHandler(common_cfg['filename'],
                                               common_cfg['filemode'])
            file_handler.setFormatter(format_str)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

            if wandb_cfg is not None and wandb is not None:
                self.enable_wandb = True
                wandb.init(**wandb_cfg)
                wandb.define_metric('global_step')
                wandb.define_metric('loss/*', step_metric='global_step')
                wandb.define_metric('eval/*', step_metric='global_step')

        global_logger = self

    def __call__(self, msg=None, level=logging.INFO, wandb_metrics=None, step=None):
        if self.rank != 0:
            return
        if msg is not None:
            self.logger.log(level, msg)

        if self.enable_wandb and wandb_metrics:
            metrics = dict(wandb_metrics)
            if step is not None:
                metrics['global_step'] = step
            wandb.log(metrics)

    def close(self):
        if self.rank == 0 and self.enable_wandb:
            wandb.finish()
