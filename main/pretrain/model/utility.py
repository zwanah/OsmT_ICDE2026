import copy
import pandas as pd
import numpy as np
import torch

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

class EpochProgressBar(TQDMProgressBar):
    """
    This extends the base progress bar to not overwrite the progress bar
    and show its history.
    """

    def on_train_epoch_end(self, trainer="pl.Trainer", pl_module="pl.LightningModule"):
        super().on_train_epoch_end(trainer=trainer, pl_module=pl_module)
        print("\n")


class HistoryCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.history.append(each_me)

    def history_dataframe(self):
        return pd.DataFrame(self.history).astype(np.float32)
    
def get_lr_scheduler(lr_scheduler,optimizer, warmup_steps, total_training_steps,final_cosine):
    if lr_scheduler =='linear':
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = warmup_steps,
                num_training_steps = total_training_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
    elif lr_scheduler =='cosine':
        scheduler1 = LinearLR(
                    optimizer,
                    start_factor=0.5,
                    end_factor=1,
                    total_iters=warmup_steps,
                    last_epoch=-1,
                )
        scheduler2 = CosineAnnealingLR(
                        optimizer,
                        T_max=total_training_steps - warmup_steps,
                        eta_min=final_cosine,
                    )
        lr_scheduler = SequentialLR(
                        optimizer,
                        schedulers=[scheduler1, scheduler2],
                        milestones=[warmup_steps]
                    )
    return lr_scheduler    

def get_optimizer(args, parameters):
    if args.adam_name == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.98), 
                                    eps=args.adam_epsilon, weight_decay=args.weight_decay)
    elif args.adam_name == "DeepSpeedCPUAdam":
        optimizer = DeepSpeedCPUAdam(parameters, adamw_mode=True, lr=args.learning_rate, betas=(0.9, 0.98),
                                    eps=args.adam_epsilon, weight_decay=args.weight_decay)
    elif args.adam_name == "FusedAdam":
        optimizer = FusedAdam(parameters, adam_w_mode=True, lr=args.learning_rate, betas=(0.9, 0.98),
                            eps=args.adam_epsilon, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Only AdamW, DeepSpeedCPUAdam, and FusedAdam are available.")
    return optimizer
