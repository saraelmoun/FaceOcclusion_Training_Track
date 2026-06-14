import os
"""Faceptor (ECCV2024, lxq1000/Faceptor) fine-tuné E2E — même protocole que les autres.

Backbone = encodeur visuel CLIP ViT-B/16 du checkpoint officiel amont
(faceptor stage_1, module.backbone_module.visual.*), pos-emb 32×32→14×14
bicubique, converti fp32 et entraîné. Tête de régression identique, norm CLIP,
fold 0, RobustOccGenderBatchSampler 64/50/14, weighted MSE, 12 epochs, 5e-6/5e-5.
"""
import argparse, json, sys, time
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import train_dinov2_large as T

CKPT = os.environ.get("FACEPTOR_CKPT", "weights/faceptor_checkpoint_rank0_iter_50000.pth.tar")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms():
    norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.10, 0.10, 0.05, 0.0)], p=0.5),
        transforms.RandomRotation(degrees=3),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, val_tf


def load_faceptor_visual():
    from clip_vit import VisionTransformer
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    pfx = "module.backbone_module.visual."
    sd = {k.removeprefix(pfx): v for k, v in ckpt.items() if k.startswith(pfx)}
    pe = sd["positional_embedding"]                 # (1025, 768) = CLS + 32×32
    cls_tok, grid = pe[:1], pe[1:]
    g = int(grid.shape[0] ** 0.5)
    grid = grid.reshape(g, g, -1).permute(2, 0, 1).unsqueeze(0)
    grid = F.interpolate(grid.float(), size=(14, 14), mode="bicubic", align_corners=False)
    grid = grid.squeeze(0).permute(1, 2, 0).reshape(196, -1).to(pe.dtype)
    sd["positional_embedding"] = torch.cat([cls_tok, grid], dim=0)  # (197, 768)
    m = VisionTransformer(input_resolution=224, patch_size=16, width=768,
                          layers=12, heads=12, output_dim=512)
    m.load_state_dict(sd, strict=True)
    return m.float()                                # CLIP fp16 -> fp32 pour l'entraînement


def build_faceptor_arch():
    """Architecture seule (sans poids officiels) — pour l'inférence, où les poids
    fine-tunés sont chargés par-dessus via load_state_dict. Pas besoin du checkpoint
    Faceptor officiel dans ce cas."""
    from clip_vit import VisionTransformer
    return VisionTransformer(input_resolution=224, patch_size=16, width=768,
                             layers=12, heads=12, output_dim=512).float()


class FaceptorRegressor(nn.Module):
    def __init__(self, dropout=0.2, pretrained=True):
        # pretrained=True (entraînement) : charge les poids Faceptor officiels.
        # pretrained=False (inférence)   : architecture seule, poids chargés ensuite.
        super().__init__()
        self.backbone = load_faceptor_visual() if pretrained else build_faceptor_arch()
        n = 512                                     # CLS projeté
        self.head = nn.Sequential(
            nn.LayerNorm(n), nn.Dropout(dropout),
            nn.Linear(n, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    name = "faceptor_b16"

    df = pd.read_csv(T.OUT_PATH)
    df_train = df[df["fold"] != T.VAL_FOLD].reset_index(drop=True)
    df_val = df[df["fold"] == T.VAL_FOLD].reset_index(drop=True)
    T.seed_everything(T.SEED)

    train_tf, val_tf = build_transforms()
    tr_ds = T.OcclusionDataset(df_train, T.IMAGE_DIR, transform=train_tf, training=True, return_meta=True)
    va_ds = T.OcclusionDataset(df_val, T.IMAGE_DIR, transform=val_tf, training=True, return_meta=True)
    bs = T.RobustOccGenderBatchSampler(
        df=df_train, batch_size=128, bin_col="occ_bin_robust", gender_col="gender",
        bin_quotas={"low_[0,0.1)": 64, "medium_[0.1,0.3)": 50, "high_[0.3,1.01)": 14},
        tau=0.75, n_batches=len(df_train) // 128, seed=T.SEED)
    tl = DataLoader(tr_ds, batch_sampler=bs, num_workers=8, pin_memory=True)
    vl = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    print("=" * 90 + "\nFaceptor CLIP ViT-B/16 (fold 0, sampler 64/50/14)\n" + "=" * 90, flush=True)
    t0 = time.time()
    model = FaceptorRegressor().to(device)
    print("[faceptor] encodeur visuel chargé (CLS projeté 512)", flush=True)
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": 5e-6},
         {"params": model.head.parameters(), "lr": 5e-5}], weight_decay=1e-4)
    nsteps = 12 * len(tl)
    sched = T.get_cosine_schedule_with_warmup(opt, int(0.05 * nsteps), nsteps, 0.05)

    best_score, best_epoch, best_pred, best_state = np.inf, -1, None, None
    history = []
    for epoch in range(1, 13):
        tl_loss = T.train_one_epoch_amp(model, tl, opt, device, sched)
        vp = T.predict_loader(model, vl, device)
        es = T.evaluate_predictions(vp, model_name=f"{name}_ep{epoch}", best_epoch=epoch, verbose=False)
        history.append({"epoch": epoch, "train_loss": tl_loss, **es})
        print(f"Epoch {epoch:02d} | train_loss={tl_loss:.6f} | score={es['val_score_global']:.6f} | "
              f"err_g0={es['err_gender_0']:.6f} | err_g1={es['err_gender_1']:.6f} | "
              f"gap={es['gender_gap']:.6f} | high_score={es['high_score']:.6f} | "
              f"low_bias={es['low_bias']:.4f} | high_bias={es['high_bias']:.4f}", flush=True)
        if np.isfinite(es["val_score_global"]) and es["val_score_global"] < best_score:
            best_score, best_epoch = es["val_score_global"], epoch
            best_pred = vp.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    runtime = time.time() - t0
    model.load_state_dict(best_state)
    summary = T.evaluate_predictions(best_pred, model_name=name, fold_runtime=runtime,
                                     best_epoch=best_epoch, verbose=True)
    print(f"\nBest epoch: {best_epoch}  Best score: {best_score:.6f}  Runtime: {runtime:.0f}s", flush=True)

    R = T.RUN_DIR
    torch.save({"model_state_dict": model.state_dict(), "summary": summary}, R / f"{name}_best.pt")
    best_pred.to_csv(R / f"val_pred_{name}.csv", index=False)
    pd.DataFrame(history).to_csv(R / f"history_{name}.csv", index=False)
    T.cell_report(best_pred, bin_col="occ_bin_fine").to_csv(R / f"report_fine_{name}.csv", index=False)
    with open(R / f"summary_{name}.json", "w") as fh:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in summary.items()}, fh, indent=1, default=str)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
