import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

from light_training.dataloading.dataset import get_train_test_loader_from_test_list
from test_list_brats2025 import test_list


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {str(k)[7:] if str(k).startswith("module") else str(k): v for k, v in sd.items()}


def metadata_name(properties):
    name = properties.get("name", "case")
    while isinstance(name, (list, tuple)) and name:
        name = name[0]
    return str(name)


def select_cases(metrics_csv, count):
    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["Avg_dice"]))
    if count >= len(rows):
        return [r["case"] for r in rows]
    idxs = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[i]["case"] for i in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/best_model_0.9027.pt")
    parser.add_argument("--out-dir", default="./prediction_results/DiffUNet_selected")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-count", type=int, default=25)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--metrics-csv", default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from diffunet.diffunet_model import DiffUNet

    cases = args.cases or select_cases(args.metrics_csv, args.case_count)
    _, ds = get_train_test_loader_from_test_list(args.data_dir, test_list)
    ds.datalist = [p for p in ds.datalist if Path(p).stem in set(cases)]

    model = DiffUNet(4, 4)
    model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    model.to(args.device)
    model.eval()
    inferer = SlidingWindowInferer(roi_size=[128, 128, 128], sw_batch_size=1, overlap=0.5, mode="gaussian")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        pending = []
        for p in ds.datalist:
            case_name = Path(p).stem
            if not (out_dir / f"{case_name}.npy").exists() or not (out_dir / f"{case_name}_prob.npz").exists():
                pending.append(p)
        ds.datalist = pending
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    for batch in tqdm(loader, total=len(loader), desc="Saving predictions"):
        image = batch["data"].float().to(args.device)
        case_name = metadata_name(batch["properties"])
        with torch.no_grad():
            output = inferer(image, lambda x: model(x, ddim=True))
            if isinstance(output, tuple):
                output = output[0]
            prob = torch.softmax(output, dim=1).cpu().numpy()[0].astype(np.float16)
            pred = np.argmax(prob, axis=0).astype(np.uint8)
        np.save(out_dir / f"{case_name}.npy", pred)
        np.savez_compressed(out_dir / f"{case_name}_prob.npz", prob=prob)

    print(f"Saved predictions to {out_dir}; pending cases processed: {len(loader)}")


if __name__ == "__main__":
    main()
