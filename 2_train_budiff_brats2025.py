"""
Training script for DiffUNet on BraTS 2025 Glioma dataset.
Adapted for RTX 3050 (8GB VRAM): batch_size=1, mixed precision friendly.
"""
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.dataloading.dataset import get_kfold_loader
import os


def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)


# ============== CONFIGURATION ==============
data_dir = "./data/fullres/train"
logdir = "./logs/diffunet"
model_save_path = os.path.join(logdir, "model")

# --- RTX 3050 adjustments ---
max_epoch = 1000
batch_size = 1           # Reduced from 2 → 1 for 8GB VRAM
val_every = 2
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]  # If OOM, change to [96, 96, 96]
augmentation = True
fold = 0                  # Which fold to use for validation (0-4)
# ================================


class BraTS2025Trainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu",
                 val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750,
                 training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device,
                         val_every, num_gpus, logdir, master_ip,
                         master_port, training_script)

        self.window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=1,       # Reduced for memory
            overlap=0.5
        )
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 4    # Reduced from 12 for CPU RAM

        from diffunet.diffunet_model import DiffUNet
        self.model = DiffUNet(4, 4)  # 4 input channels, 4 output classes

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-2,
            weight_decay=3e-5,
            momentum=0.99,
            nesterov=True
        )
        self.scheduler_type = "poly"
        self.loss_func = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label = self.get_input(batch)

        pred, pred_edge, uncertainty = self.model(image, label)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0)

        loss = self.loss_func(pred, label)
        loss_edge = self.loss_func(pred_edge, label)

        scale = func(self.epoch, max_epoch)
        loss = loss.mean() + loss_edge.mean() + (loss * uncertainty).mean() * scale

        self.log("training_loss", loss.mean(), step=self.global_step)
        self.log("training_loss_edge", loss_edge.mean(), step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        label = label[:, 0].long()  # (B, D, H, W) with class indices 0-3
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
        output = output.argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []
        c = 4
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i
            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs
        dices_mean = []
        c = 3
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)

        self.log("class1_NCR_NET", dices_mean[0], step=self.epoch)
        self.log("class2_ED", dices_mean[1], step=self.epoch)
        self.log("class3_ET", dices_mean[2], step=self.epoch)
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

        print(f"Epoch {self.epoch}: mean_dice = {mean_dice:.4f} | "
              f"NCR/NET={dices_mean[0]:.4f}, ED={dices_mean[1]:.4f}, ET={dices_mean[2]:.4f}")


if __name__ == "__main__":
    os.makedirs(model_save_path, exist_ok=True)

    trainer = BraTS2025Trainer(
        env_type="pytorch",
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        master_port=17753,
        training_script=__file__
    )

    # Use 5-fold CV: fold=0 means 80% train, 20% validation
    train_ds, val_ds, _ = get_kfold_loader(data_dir=data_dir, fold=fold)

    print(f"Training samples  : {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Patch size: {patch_size}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epoch}")

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)