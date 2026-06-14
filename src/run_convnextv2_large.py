"""ConvNeXt V2-Large sur le MÊME protocole que le run canonique DINOv2-large :
même split (fold 0, StratifiedGroupKFold par identité), même RobustOccGenderBatchSampler
(batch 128, quotas 64/50/14, tau 0.75, seed 42), même weighted MSE, 12 epochs,
warmup 5 % + cosine. Seuls le backbone et les LR changent (gros convnet :
lr_backbone 2e-5 / lr_head 2e-4, entre le convnext_tiny du notebook et le ViT-L).

Tourne sur cuda:1 pendant que le DINOv2 occupe cuda:0.
"""
import json
import sys

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import train_dinov2_large as T

RUN_DIR = T.RUN_DIR
NAME = "convnextv2_large_myfold"
BACKBONE = "convnextv2_large.fcmae_ft_in22k_in1k"


def main():
    device = torch.device("cuda:1")
    df = pd.read_csv(T.OUT_PATH)  # folds déjà construits par le run DINOv2
    df_train = df[df["fold"] != T.VAL_FOLD].reset_index(drop=True)
    df_val = df[df["fold"] == T.VAL_FOLD].reset_index(drop=True)
    print(f"train {len(df_train)}  val {len(df_val)}", flush=True)

    T.seed_everything(T.SEED)
    BATCH_SIZE, TAU = 128, 0.75
    bin_quotas = {"low_[0,0.1)": 64, "medium_[0.1,0.3)": 50, "high_[0.3,1.01)": 14}

    train_dataset = T.OcclusionDataset(df_train, T.IMAGE_DIR,
                                       transform=T.train_transform,
                                       training=True, return_meta=True)
    val_dataset = T.OcclusionDataset(df_val, T.IMAGE_DIR,
                                     transform=T.val_transform,
                                     training=True, return_meta=True)
    batch_sampler = T.RobustOccGenderBatchSampler(
        df=df_train, batch_size=BATCH_SIZE, bin_col="occ_bin_robust",
        gender_col="gender", bin_quotas=bin_quotas, tau=TAU,
        n_batches=len(df_train) // BATCH_SIZE, seed=T.SEED)
    print(batch_sampler.get_quota_table().to_string(index=False), flush=True)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                              num_workers=8, pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=False,
                            persistent_workers=False)

    T.seed_everything(T.SEED)
    model, best_pred, history, summary = T.run_one_fold_large_stable(
        model_name=NAME, backbone_name=BACKBONE,
        train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=12, lr_backbone=2e-5, lr_head=2e-4,
        weight_decay=1e-4, dropout=0.2, warmup_ratio=0.05, min_lr_ratio=0.05)

    torch.save({"model_state_dict": model.state_dict(),
                "backbone_name": BACKBONE, "val_fold": T.VAL_FOLD,
                "summary": summary}, RUN_DIR / f"{NAME}_best.pt")
    best_pred.to_csv(RUN_DIR / f"val_pred_{NAME}.csv", index=False)
    history.to_csv(RUN_DIR / f"history_{NAME}.csv", index=False)
    T.cell_report(best_pred, bin_col="occ_bin_fine").to_csv(
        RUN_DIR / f"report_fine_{NAME}.csv", index=False)
    T.cell_report(best_pred, bin_col="occ_bin_robust").to_csv(
        RUN_DIR / f"report_robust_{NAME}.csv", index=False)
    with open(RUN_DIR / f"summary_{NAME}.json", "w") as fh:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in summary.items()}, fh, indent=1, default=str)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
