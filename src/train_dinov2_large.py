"""Reproduction fidèle du run canonique DINOv2-large du notebook DINO-Copy4 (1).ipynb.

Cellules reproduites à l'identique : split StratifiedGroupKFold par identité
(cells 2-30), dataset/transforms (40-47), RobustOccGenderBatchSampler (49-53, 96),
métrique officielle + cell_report (68-71, 85), boucle d'entraînement AMP bf16
(81-88) et run canonique cell 97 :
  vit_large_patch14_dinov2.lvd142m, 12 epochs, lr_backbone 5e-6, lr_head 5e-5,
  wd 1e-4, dropout 0.2, warmup_ratio 0.05, min_lr_ratio 0.05, fold de val 0.

Sorties (/workspace/dino_run/) :
  train_with_folds.csv, dinov2_fold0_best.pt, val_pred_fold0.csv,
  history_fold0.csv, report_fine.csv, report_robust.csv, summary.json
"""
from __future__ import annotations
import os

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

# Chemins paramétrables (défauts = racine du repo) → portable pour un tiers.
_REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace"))
CSV_PATH = Path(os.environ.get("TRAIN_CSV", DATA_ROOT / "occlusion_datasets" / "train.csv"))
OUT_PATH = Path(os.environ.get("FOLDS_CSV", _REPO / "data" / "train_with_folds.csv"))
IMAGE_DIR = Path(os.environ.get("CROPS_DIR", "/root/crops/Crop_224_5fp_100K"))
RUN_DIR = Path(os.environ.get("OUTDIR", _REPO / "weights"))

SEED = 42
N_SPLITS = 5
VAL_FOLD = 0


# ───────────────────────── split (cells 2-30) ─────────────────────────
def extract_identity(filename: str) -> str:
    path = str(filename).replace("\\", "/")
    parts = path.split("/")
    m_parts = [p for p in parts if p.startswith("m.")]
    if len(m_parts) > 0:
        return m_parts[-1]
    stem_path = str(Path(path).with_suffix("")).replace("/", "__")
    return f"pseudo__{stem_path}"


def extract_identity_type(filename: str) -> str:
    parts = str(filename).replace("\\", "/").split("/")
    return ("real_identity_from_path" if any(p.startswith("m.") for p in parts)
            else "pseudo_identity_from_filename")


def build_folds() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["FaceOcclusion"] = df["FaceOcclusion"].astype(float)
    df["gender"] = df["gender"].astype(float).astype(int)
    df["identity"] = df["filename"].apply(extract_identity)
    df["identity_type"] = df["filename"].apply(extract_identity_type)
    df["source"] = (df["filename"].astype(str).str.replace("\\", "/", regex=False)
                    .str.split("/").str[0])

    fine_bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.01]
    fine_labels = ["[0,0.05)", "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.3)",
                   "[0.3,0.4)", "[0.4,1.01)"]
    df["occ_bin_fine"] = pd.cut(df["FaceOcclusion"], bins=fine_bins,
                                labels=fine_labels, right=False, include_lowest=True)
    robust_bins = [0.0, 0.1, 0.3, 1.01]
    robust_labels = ["low_[0,0.1)", "medium_[0.1,0.3)", "high_[0.3,1.01)"]
    df["occ_bin_robust"] = pd.cut(df["FaceOcclusion"], bins=robust_bins,
                                  labels=robust_labels, right=False, include_lowest=True)
    assert df["occ_bin_fine"].isna().sum() == 0
    assert df["occ_bin_robust"].isna().sum() == 0

    df["cell_fine"] = df["gender"].astype(str) + "__" + df["occ_bin_fine"].astype(str)
    df["metric_weight"] = 1 / 30 + df["FaceOcclusion"]

    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(
            sgkf.split(df, df["cell_fine"].astype(str), df["identity"].astype(str))):
        df.loc[val_idx, "fold"] = fold
    assert (df["fold"] >= 0).all()
    df.to_csv(OUT_PATH, index=False)
    print("folds:", df["fold"].value_counts().sort_index().to_dict(), flush=True)
    return df


# ─────────────────── seeds / transforms (cells 40-42) ───────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.10, contrast=0.10,
                                saturation=0.05, hue=0.0)], p=0.5),
    transforms.RandomRotation(degrees=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ───────────────────── dataset / sampler (cells 45-53) ─────────────────────
class OcclusionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, training=True, return_meta=True):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.training = training
        self.return_meta = return_meta
        self.filenames = self.df["filename"].tolist()
        self.paths = [self.image_dir / f for f in self.filenames]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if not self.training:
            return {"image": img, "filename": row["filename"]}
        output = {
            "image": img,
            "target": torch.tensor(row["FaceOcclusion"], dtype=torch.float32),
            "gender": torch.tensor(row["gender"], dtype=torch.float32),
            "filename": row["filename"],
        }
        if self.return_meta:
            output.update({
                "metric_weight": torch.tensor(
                    row.get("metric_weight", 1 / 30 + row["FaceOcclusion"]),
                    dtype=torch.float32),
                "identity": row.get("identity", ""),
                "identity_type": row.get("identity_type", ""),
                "source": row.get("source", ""),
                "fold": int(row["fold"]) if "fold" in row else -1,
                "occ_bin_fine": str(row.get("occ_bin_fine", "")),
                "occ_bin_robust": str(row.get("occ_bin_robust", "")),
            })
        return output


class RobustOccGenderBatchSampler(Sampler):
    def __init__(self, df, batch_size=128, bin_col="occ_bin_robust",
                 gender_col="gender", bin_quotas=None, tau=0.5,
                 n_batches=None, seed=42):
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = batch_size
        self.bin_col = bin_col
        self.gender_col = gender_col
        self.tau = tau
        self.seed = seed
        if bin_quotas is None:
            bin_quotas = {"low_[0,0.1)": 48, "medium_[0.1,0.3)": 48,
                          "high_[0.3,1.01)": 32}
        self.bin_quotas = bin_quotas
        assert sum(self.bin_quotas.values()) == self.batch_size
        if n_batches is None:
            n_batches = len(self.df) // self.batch_size
        self.n_batches = int(n_batches)
        self.cell_indices = self._build_cell_indices()
        self.quotas = self._build_quotas()
        self._check_quotas()

    def _build_cell_indices(self):
        cell_indices = {}
        for bin_name in self.bin_quotas.keys():
            for gender_value in sorted(self.df[self.gender_col].unique()):
                mask = (self.df[self.bin_col].astype(str).eq(str(bin_name))
                        & self.df[self.gender_col].eq(gender_value))
                indices = np.where(mask.values)[0]
                if len(indices) > 0:
                    cell_indices[(str(bin_name), gender_value)] = indices
        return cell_indices

    def _build_quotas(self):
        quotas = {}
        counts = (self.df.groupby([self.bin_col, self.gender_col], observed=False)
                  .size().rename("n_available").reset_index())
        for bin_name, bin_quota in self.bin_quotas.items():
            sub = counts[counts[self.bin_col].astype(str).eq(str(bin_name))].copy()
            sub = sub[sub["n_available"] > 0].copy()
            if len(sub) == 0:
                raise ValueError(f"Aucun exemple disponible pour bin {bin_name}")
            raw = sub["n_available"].astype(float).values ** self.tau
            proportions = raw / raw.sum()
            exact = proportions * bin_quota
            floor = np.floor(exact).astype(int)
            remainder = bin_quota - floor.sum()
            frac = exact - floor
            order = np.argsort(-frac)
            final = floor.copy()
            for i in order[:remainder]:
                final[i] += 1
            for (_, row), q in zip(sub.iterrows(), final):
                quotas[(str(bin_name), row[self.gender_col])] = int(q)
        return quotas

    def _check_quotas(self):
        assert sum(self.quotas.values()) == self.batch_size
        for key, q in self.quotas.items():
            if q <= 0:
                raise ValueError(f"Quota nul pour {key}: {q}")
            if key not in self.cell_indices:
                raise ValueError(f"Aucune donnée disponible pour cellule {key}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_batches):
            batch = []
            for key, q in self.quotas.items():
                batch.extend(rng.choice(self.cell_indices[key], size=q,
                                        replace=True).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches

    def get_quota_table(self):
        rows = []
        for (bin_name, gender_value), q in self.quotas.items():
            n_av = len(self.cell_indices[(bin_name, gender_value)])
            rows.append({"occ_bin_robust": bin_name, "gender": gender_value,
                         "n_available": n_av, "n_per_batch": q,
                         "n_batches": self.n_batches,
                         "expected_per_epoch": q * self.n_batches,
                         "repeat_factor": q * self.n_batches / n_av})
        return pd.DataFrame(rows).sort_values(["occ_bin_robust", "gender"])


# ───────────────────── métrique / rapports (cells 68-85) ─────────────────────
def weighted_err_for_gender(df_pred, gender_value, pred_col="pred",
                            target_col="FaceOcclusion"):
    sub = df_pred[df_pred["gender"] == gender_value].copy()
    if len(sub) == 0:
        return np.nan
    y = sub[target_col].astype(float).values
    p = sub[pred_col].astype(float).values
    w = 1 / 30 + y
    return np.sum(w * (p - y) ** 2) / np.sum(w)


def safe_official_score(df_pred, pred_col="pred", target_col="FaceOcclusion"):
    d = df_pred.copy()
    out = {"score": np.nan, "err_gender_0": np.nan, "err_gender_1": np.nan,
           "gap": np.nan, "worst_gender": np.nan, "ratio_worst_best": np.nan}
    if len(d) == 0:
        return out
    genders_present = sorted(d["gender"].astype(int).unique())
    if 0 in genders_present:
        out["err_gender_0"] = weighted_err_for_gender(d, 0, pred_col, target_col)
    if 1 in genders_present:
        out["err_gender_1"] = weighted_err_for_gender(d, 1, pred_col, target_col)
    if 0 in genders_present and 1 in genders_present:
        e0, e1 = out["err_gender_0"], out["err_gender_1"]
        out["gap"] = abs(e0 - e1)
        out["score"] = (e0 + e1) / 2 + out["gap"]
        out["worst_gender"] = 0 if e0 > e1 else 1
        out["ratio_worst_best"] = max(e0, e1) / max(min(e0, e1), 1e-12)
    return out


def subset_metrics(df_pred, mask, prefix, pred_col="pred",
                   target_col="FaceOcclusion"):
    sub = df_pred[mask].copy()
    if len(sub) == 0:
        return {f"{prefix}_n": 0, f"{prefix}_score": np.nan,
                f"{prefix}_err_gender_0": np.nan, f"{prefix}_err_gender_1": np.nan,
                f"{prefix}_gap": np.nan, f"{prefix}_bias": np.nan,
                f"{prefix}_mean_y": np.nan, f"{prefix}_mean_pred": np.nan}
    s = safe_official_score(sub, pred_col, target_col)
    return {f"{prefix}_n": len(sub), f"{prefix}_score": s["score"],
            f"{prefix}_err_gender_0": s["err_gender_0"],
            f"{prefix}_err_gender_1": s["err_gender_1"],
            f"{prefix}_gap": s["gap"],
            f"{prefix}_bias": float((sub[pred_col] - sub[target_col]).mean()),
            f"{prefix}_mean_y": float(sub[target_col].mean()),
            f"{prefix}_mean_pred": float(sub[pred_col].mean())}


def cell_report(df_pred, pred_col="pred", target_col="FaceOcclusion",
                bin_col="occ_bin_fine"):
    d = df_pred.copy()
    d["metric_weight_eval"] = 1 / 30 + d[target_col].astype(float)
    d["sq_error"] = (d[pred_col].astype(float) - d[target_col].astype(float)) ** 2
    d["weighted_sq_error"] = d["metric_weight_eval"] * d["sq_error"]
    d["bias"] = d[pred_col].astype(float) - d[target_col].astype(float)
    report = (d.groupby(["gender", bin_col], observed=False)
              .agg(n=(target_col, "size"), mean_y=(target_col, "mean"),
                   mean_pred=(pred_col, "mean"), bias=("bias", "mean"),
                   weight_sum=("metric_weight_eval", "sum"),
                   weighted_error_num=("weighted_sq_error", "sum"))
              .reset_index())
    report["weighted_error"] = report["weighted_error_num"] / report["weight_sum"]
    return report.drop(columns=["weighted_error_num"]).sort_values(["gender", bin_col])


def evaluate_predictions(df_pred, model_name="model", pred_col="pred",
                         target_col="FaceOcclusion", fold_runtime=np.nan,
                         best_epoch=np.nan, verbose=True):
    d = df_pred.copy()
    d[pred_col] = d[pred_col].astype(float).clip(0.0, 1.0)
    d[target_col] = d[target_col].astype(float)
    d["gender"] = d["gender"].astype(int)
    g = safe_official_score(d, pred_col, target_col)
    low = subset_metrics(d, d["occ_bin_robust"].astype(str).eq("low_[0,0.1)"), "low")
    medium = subset_metrics(d, d["occ_bin_robust"].astype(str).eq("medium_[0.1,0.3)"), "medium")
    high = subset_metrics(d, d["occ_bin_robust"].astype(str).eq("high_[0.3,1.01)"), "high")
    very_high = subset_metrics(d, d["occ_bin_fine"].astype(str).eq("[0.4,1.01)"), "very_high")
    summary = {
        "model": model_name,
        "val_score_global": g["score"], "err_gender_0": g["err_gender_0"],
        "err_gender_1": g["err_gender_1"], "gender_gap": g["gap"],
        "worst_gender": g["worst_gender"], "ratio_worst_best": g["ratio_worst_best"],
        "high_score": high["high_score"],
        "high_err_gender_0": high["high_err_gender_0"],
        "high_err_gender_1": high["high_err_gender_1"], "high_gap": high["high_gap"],
        "low_bias": low["low_bias"], "medium_bias": medium["medium_bias"],
        "high_bias": high["high_bias"],
        "very_high_score": very_high["very_high_score"],
        "very_high_err_gender_0": very_high["very_high_err_gender_0"],
        "very_high_err_gender_1": very_high["very_high_err_gender_1"],
        "very_high_gap": very_high["very_high_gap"],
        "very_high_bias": very_high["very_high_bias"],
        "fold_runtime": fold_runtime, "best_epoch": best_epoch,
        "n_val": len(d), "n_low": low["low_n"], "n_medium": medium["medium_n"],
        "n_high": high["high_n"], "n_very_high": very_high["very_high_n"],
    }
    if verbose:
        print("=" * 100 + f"\n{model_name}\n" + "=" * 100, flush=True)
        print(pd.DataFrame([summary]).T.to_string(), flush=True)
        print("\nReport occ_bin_robust", flush=True)
        print(cell_report(d, bin_col="occ_bin_robust").to_string(index=False), flush=True)
        print("\nReport occ_bin_fine", flush=True)
        print(cell_report(d, bin_col="occ_bin_fine").to_string(index=False), flush=True)
    return summary


# ─────────────────── modèle / entraînement (cells 81-93) ───────────────────
def weighted_mse_loss(pred, target, weight):
    pred, target, weight = pred.view(-1), target.view(-1), weight.view(-1)
    return (weight * (pred - target) ** 2).sum() / weight.sum()


class RegressionModel(nn.Module):
    def __init__(self, backbone_name, pretrained=True, dropout=0.2, img_size=224):
        super().__init__()
        is_vit = any(k in backbone_name for k in ("vit", "dinov2", "patch14"))
        kw = {"global_pool": "token" if is_vit else "avg"}
        if "patch14" in backbone_name:
            kw["img_size"] = img_size
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, **kw)
        n = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(n), nn.Dropout(dropout),
            nn.Linear(n, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, min_lr_ratio=0.05):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch_amp(model, train_loader, optimizer, device,
                        scheduler=None, max_batches=None):
    model.train()
    losses = []
    use_amp = device.type == "cuda"
    for step, batch in enumerate(tqdm(train_loader, desc="Train", leave=False,
                                      mininterval=30)):
        if max_batches is not None and step >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        weights = batch["metric_weight"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            preds = model(images)
            loss = weighted_mse_loss(preds, targets, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict_loader(model, loader, device, max_batches=None, clip_pred=True):
    model.eval()
    rows = []
    for step, batch in enumerate(tqdm(loader, desc="Predict", leave=False,
                                      mininterval=30)):
        if max_batches is not None and step >= max_batches:
            break
        images = batch["image"].to(device)
        preds = model(images).detach().cpu().numpy()
        for i in range(len(preds)):
            rows.append({
                "filename": batch["filename"][i], "pred": float(preds[i]),
                "FaceOcclusion": float(batch["target"][i].item()),
                "gender": int(batch["gender"][i].item()),
                "fold": int(batch["fold"][i]), "identity": batch["identity"][i],
                "source": batch["source"][i],
                "identity_type": batch["identity_type"][i],
                "occ_bin_fine": batch["occ_bin_fine"][i],
                "occ_bin_robust": batch["occ_bin_robust"][i]})
    df_pred = pd.DataFrame(rows)
    if clip_pred:
        df_pred["pred"] = df_pred["pred"].clip(0.0, 1.0)
    return df_pred


def run_one_fold_large_stable(model_name, backbone_name, train_loader, val_loader,
                              device, epochs=10, lr_backbone=5e-6, lr_head=1e-4,
                              weight_decay=1e-4, dropout=0.2, pretrained=True,
                              warmup_ratio=0.10, min_lr_ratio=0.05):
    print("=" * 100 + f"\nMODEL: {model_name}\nBACKBONE: {backbone_name}\n" + "=" * 100,
          flush=True)
    start_time = time.time()
    model = RegressionModel(backbone_name=backbone_name, pretrained=pretrained,
                            dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": lr_backbone},
         {"params": model.head.parameters(), "lr": lr_head}],
        weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr_ratio)
    print(f"Training steps: {num_training_steps}  Warmup: {num_warmup_steps}  "
          f"LR bb/head: {lr_backbone}/{lr_head}", flush=True)

    best_score, best_epoch, best_pred, best_state = np.inf, -1, None, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_amp(model, train_loader, optimizer, device,
                                         scheduler)
        val_pred = predict_loader(model, val_loader, device)
        es = evaluate_predictions(val_pred, model_name=f"{model_name}_epoch_{epoch}",
                                  best_epoch=epoch, verbose=False)
        score = es["val_score_global"]
        history.append({"epoch": epoch, "train_loss": train_loss, **es})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
              f"score={es['val_score_global']:.6f} | "
              f"err_g0={es['err_gender_0']:.6f} | err_g1={es['err_gender_1']:.6f} | "
              f"gap={es['gender_gap']:.6f} | high_score={es['high_score']:.6f} | "
              f"low_bias={es['low_bias']:.4f} | medium_bias={es['medium_bias']:.4f} | "
              f"high_bias={es['high_bias']:.4f}", flush=True)
        if np.isfinite(score) and score < best_score:
            best_score, best_epoch = score, epoch
            best_pred = val_pred.copy()
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    fold_runtime = time.time() - start_time
    if best_pred is None:
        raise RuntimeError("Aucune prédiction valide : score NaN/inf.")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    final_summary = evaluate_predictions(best_pred, model_name=model_name,
                                         fold_runtime=fold_runtime,
                                         best_epoch=best_epoch, verbose=True)
    print(f"\nBest epoch: {best_epoch}  Best score: {best_score:.6f}  "
          f"Runtime: {fold_runtime:.0f}s", flush=True)
    return model, best_pred, pd.DataFrame(history), final_summary


def main():
    device = torch.device("cuda:0")
    df = build_folds()
    df_train = df[df["fold"] != VAL_FOLD].reset_index(drop=True)
    df_val = df[df["fold"] == VAL_FOLD].reset_index(drop=True)
    print(f"train {len(df_train)}  val {len(df_val)}", flush=True)

    seed_everything(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    # CONFIG cell 51 + loaders "safe" cells 89/96 (run canonique)
    BATCH_SIZE, TAU = 128, 0.75
    bin_quotas = {"low_[0,0.1)": 64, "medium_[0.1,0.3)": 50, "high_[0.3,1.01)": 14}
    NUM_WORKERS_SAFE = 8

    train_dataset = OcclusionDataset(df_train, IMAGE_DIR, transform=train_transform,
                                     training=True, return_meta=True)
    val_dataset = OcclusionDataset(df_val, IMAGE_DIR, transform=val_transform,
                                   training=True, return_meta=True)
    batch_sampler = RobustOccGenderBatchSampler(
        df=df_train, batch_size=BATCH_SIZE, bin_col="occ_bin_robust",
        gender_col="gender", bin_quotas=bin_quotas, tau=TAU,
        n_batches=len(df_train) // BATCH_SIZE, seed=SEED)
    print(batch_sampler.get_quota_table().to_string(index=False), flush=True)

    pin_memory = True
    train_loader_safe = DataLoader(train_dataset, batch_sampler=batch_sampler,
                                   num_workers=NUM_WORKERS_SAFE,
                                   pin_memory=pin_memory, persistent_workers=False)
    val_loader_safe = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS_SAFE, pin_memory=pin_memory,
                                 drop_last=False, persistent_workers=False)

    seed_everything(SEED)
    model, best_pred, history, summary = run_one_fold_large_stable(
        model_name="dinov2_large_stable_12epoch",
        backbone_name="vit_large_patch14_dinov2.lvd142m",
        train_loader=train_loader_safe, val_loader=val_loader_safe,
        device=device, epochs=12, lr_backbone=5e-6, lr_head=5e-5,
        weight_decay=1e-4, dropout=0.2, warmup_ratio=0.05, min_lr_ratio=0.05)

    torch.save({"model_state_dict": model.state_dict(),
                "backbone_name": "vit_large_patch14_dinov2.lvd142m",
                "val_fold": VAL_FOLD, "summary": summary},
               RUN_DIR / "dinov2_fold0_best.pt")
    best_pred.to_csv(RUN_DIR / "val_pred_fold0.csv", index=False)
    history.to_csv(RUN_DIR / "history_fold0.csv", index=False)
    cell_report(best_pred, bin_col="occ_bin_fine").to_csv(
        RUN_DIR / "report_fine.csv", index=False)
    cell_report(best_pred, bin_col="occ_bin_robust").to_csv(
        RUN_DIR / "report_robust.csv", index=False)
    with open(RUN_DIR / "summary.json", "w") as fh:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in summary.items()}, fh, indent=1, default=str)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
