# 3_train_diffunet_brats2025.py
import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
import os

def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)

# ============================================================
# CONFIGURATION — ADJUSTED FOR RTX 3050 (8GB)
# ============================================================
data_dir = "./data/fullres/train"
fold = 0
logdir = "./logs/diffunet"
env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 500
batch_size = 1          # RTX 3050: MUST be 1
val_every = 10
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]  # if OOM, change to [96,96,96]
augmentation = True
# ============================================================


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1,
                 num_gpus=1, logdir="./logs/", master_ip='localhost',
                 master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=1,
            overlap=0.5
        )
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 2

        from diffunet.diffunet_model import DiffUNet
        self.model = DiffUNet(4, 4)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=5e-3, weight_decay=3e-5,
            momentum=0.99, nesterov=True
        )
        self.scheduler_type = "poly"
        self.loss_func = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image, label)

        # ============================================================
        # Handle both cases: model returns 1 value or 3 values
        # ============================================================
        if isinstance(output, tuple) and len(output) == 3:
            # Full version: returns (pred, pred_edge, uncertainty)
            pred, pred_edge, uncertainty = output
            uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

            loss = self.loss_func(pred, label)
            loss_edge = self.loss_func(pred_edge, label)
            scale = func(self.epoch, max_epoch)

            loss = loss.mean() + loss_edge.mean() + (loss * uncertainty).mean() * scale

            self.log("training_loss", loss.mean(), step=self.global_step)
            self.log("training_loss_edge", loss_edge.mean(), step=self.global_step)

        elif isinstance(output, tuple) and len(output) == 2:
            # Partial version: returns (pred, something_else)
            pred = output[0]
            loss = self.loss_func(pred, label)
            loss = loss.mean()

            self.log("training_loss", loss, step=self.global_step)

        else:
            # Simplified version: returns only pred
            pred = output
            loss = self.loss_func(pred, label)
            loss = loss.mean()

            self.log("training_loss", loss, step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image, ddim=True)

        # Handle both return formats
        if isinstance(output, tuple):
            output = output[0]

        output = output.argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()

        # BraTS evaluation: WT, TC, ET
        dices = []

        # WT (Whole Tumor): labels 1+2+3
        pred_wt = (output > 0)
        target_wt = (target > 0)
        d_wt, _ = self.cal_metric(target_wt, pred_wt)
        dices.append(d_wt)

        # TC (Tumor Core): labels 1+3
        pred_tc = (output == 1) | (output == 3)
        target_tc = (target == 1) | (target == 3)
        d_tc, _ = self.cal_metric(target_tc, pred_tc)
        dices.append(d_tc)

        # ET (Enhancing Tumor): label 3
        pred_et = (output == 3)
        target_et = (target == 3)
        d_et, _ = self.cal_metric(target_et, pred_et)
        dices.append(d_et)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs
        d_wt = dices[0].mean()
        d_tc = dices[1].mean()
        d_et = dices[2].mean()
        mean_dice = (d_wt + d_tc + d_et) / 3.0

        self.log("WT_dice", d_wt, step=self.epoch)
        self.log("TC_dice", d_tc, step=self.epoch)
        self.log("ET_dice", d_et, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )

        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        print(f"Epoch {self.epoch}: WT={d_wt:.4f}, TC={d_tc:.4f}, ET={d_et:.4f}, Mean={mean_dice:.4f}")


if __name__ == "__main__":
    trainer = BraTSTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17753,
        training_script=__file__
    )

    from test_list_brats2025 import test_list
    train_ds, val_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)