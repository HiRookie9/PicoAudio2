from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map
from omegaconf import OmegaConf, DictConfig
from trainer import Trainer
from utils.log_helper import LoggingLogger


@dataclass(kw_only=True)
class AudioGenerationTrainer(Trainer):
    """
    Trainer for audio generation models, supporting distributed training,
    logging, validation, and loss calculation.

    Args:
        logging_file (str | Path): Path to log file for training logs.

    Attributes:
        logger: LoggingLogger instance for logging training info.
        train_loss: Accumulated training loss for current epoch.
        train_batch_num: Number of training batches in current epoch.
        val_loss_dict: Dictionary to accumulate validation losses.
        val_batch_num: Number of validation batches in current epoch.
    """


    logging_file: str | Path

    def on_train_start(self):
        """
        Called at the start of training.
        Initializes logging, saves config, and logs parameter counts.
        """
        super().on_train_start()
        self.train_loss = 0
        self.train_batch_num = 0
        if self.accelerator.is_main_process:
            self.logger = LoggingLogger(self.logging_file).create_instance()
            with open(self.project_dir / "config.yaml", "w") as writer:
                OmegaConf.save(self.config_dict, writer)

        if isinstance(self.model, DistributedDataParallel):
            num_params, trainable_params = self.model.module.count_params()
        else:
            num_params, trainable_params = self.model.count_params()

        if self.accelerator.is_main_process:
            self.logger.info(
                f"parameter number: {num_params}, trainable parameter number: {trainable_params}"
            )

    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_batch_num = 0

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch: Input batch data.
            batch_idx: Batch index.

        Returns:
            loss (Tensor): Diffusion loss tensor for backpropagation.
        """
        output = self.model(**batch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        loss_dict = self.loss_fn(output)
        log_dict = {}
        for loss_name in loss_dict:
            if loss_name != "loss":
                log_dict[f"train/{loss_name}"] = loss_dict[loss_name].item()
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        #print(loss_dict)
        loss = loss_dict["loss"]['diff_loss']
        self.train_loss += loss.item()
        self.train_batch_num += 1
        return loss

    def on_validation_start(self):
        self.val_loss_dict = defaultdict(int)
        self.val_batch_num = 0

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        output = self.accelerator.gather_for_metrics(output)
        output = tree_map(lambda x: x.mean(), output)
        loss_dict = self.loss_fn(output)
        loss = loss_dict["loss"]['diff_loss']
        #print(loss)
        self.val_loss_dict['diff_loss'] += loss.item()
        self.val_batch_num += 1

    def get_val_metrics(self):
        """
        Calculates and returns averaged validation metrics.

        Returns:
            dict: Averaged validation loss.
        """
        return {"loss": self.val_loss_dict["diff_loss"] / self.val_batch_num}

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Logs averaged train and validation losses.
        """
        train_loss = self.train_loss / self.train_batch_num
        #print(self.val_loss_dict["diff_loss"])
        val_loss = self.val_loss_dict["diff_loss"]  / self.val_batch_num
        log_dict = {}
       
        log_dict[f"val/diff_loss"] = self.val_loss_dict["diff_loss"] / self.val_batch_num
        self.accelerator.log(
            log_dict,
            step=self.step,
        )
        logging_msg = f"epoch[{self.epoch}], train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
        self.accelerator.print(logging_msg)
        if self.accelerator.is_main_process:
            self.logger.info(logging_msg)
