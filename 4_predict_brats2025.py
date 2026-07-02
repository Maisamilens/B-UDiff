import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice as dice_metric
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import SimpleITK as sitk
import os

# ============================================================
# CONFIGURATION
# ============================================================
data_dir = "./data/fullres/train"
env = "pytorch"
max_epoch = 500
batch_size = 1
val_every = 10
num_gpus = 1
device = "cuda:0"
patch_size = [128, 128, 128]

MODEL_PATH = r"./logs/diffunet/model/final_model_0.8973.pt"
# ============================================================


class BraTSPredictor(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        self.augmentation = False

        from diffunet.diffunet_model import DiffUNet

        self.model = DiffUNet(4, 4)

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        if "module" in ckpt:
            ckpt = ckpt["module"]

        new_sd = {}
        for k, v in ckpt.items():
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v

        self.model.load_state_dict(new_sd)
        self.model.to(device)
        self.model.eval()

        self.inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
            progress=True
        )

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]

        label = label[:, 0].long()
        return image, label, properties

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)

        with torch.no_grad():
            output = self.inferer(
                image.to(device),
                lambda x: self.model(x, ddim=True)
            )

            if isinstance(output, tuple):
                output = output[0]

            output = torch.argmax(output, dim=1).cpu().numpy()
            target = label.cpu().numpy()

        def safe_dice(pred, gt):
            pred = pred.astype(np.bool_)
            gt = gt.astype(np.bool_)

            if pred.sum() == 0 and gt.sum() == 0:
                return 1.0
            if pred.sum() == 0 or gt.sum() == 0:
                return 0.0
            return float(dice_metric(pred, gt))

        def compute(name, fn):
            pred_c = fn(output[0])
            target_c = fn(target[0])
            return safe_dice(pred_c, target_c)

        wt = compute("WT", lambda o: o > 0)
        tc = compute("TC", lambda o: (o == 1) | (o == 3))
        et = compute("ET", lambda o: o == 3)

        print(
            f"Case {properties['name'][0]}: "
            f"WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f}"
        )

        return wt, tc, et


if __name__ == "__main__":

    trainer = BraTSPredictor(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        val_every=val_every,
        num_gpus=num_gpus,
        logdir="",
        master_port=17751,
        training_script=__file__
    )

    from test_list_brats2025 import test_list

    train_ds, val_ds = get_train_test_loader_from_test_list(
        data_dir=data_dir,
        test_list=test_list
    )

    results = trainer.validation_single_gpu(val_ds)

    # ============================================================
    # 🔥 FIXED SAFE AGGREGATION (NO NUMPY ARRAY CONVERSION ERROR)
    # ============================================================

    wt_scores = []
    tc_scores = []
    et_scores = []

    for r in results:
        # force clean float conversion
        wt_scores.append(float(r[0]))
        tc_scores.append(float(r[1]))
        et_scores.append(float(r[2]))

    wt_scores = np.array(wt_scores, dtype=np.float32)
    tc_scores = np.array(tc_scores, dtype=np.float32)
    et_scores = np.array(et_scores, dtype=np.float32)

    print("\n=== Final Validation Metrics ===")
    print(f"WT Dice: {wt_scores.mean():.4f}")
    print(f"TC Dice: {tc_scores.mean():.4f}")
    print(f"ET Dice: {et_scores.mean():.4f}")
    print(f"Mean Dice: {(wt_scores.mean() + tc_scores.mean() + et_scores.mean()) / 3:.4f}")