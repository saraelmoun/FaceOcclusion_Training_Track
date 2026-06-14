import os
"""Inférence depuis les poids : 15 checkpoints → prédictions test (avec TTA miroir).

Reproduit les CSV de predictions/testpred_*.csv à partir des poids (weights/).
Permet la reproduction COMPLÈTE depuis les poids (et non seulement depuis les CSV).

Usage :
  python inference.py --weights_dir weights --out predictions
  python assemble.py        # puis : prédictions → soumission
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import train_dinov2_large as T
from run_faceptor_e2e import FaceptorRegressor, CLIP_MEAN, CLIP_STD

CROPS = Path(os.environ.get("CROPS_DIR","/root/crops/Crop_224_5fp_100K"))
IMN = ([0.485,0.456,0.406],[0.229,0.224,0.225]); CLP = (list(CLIP_MEAN),list(CLIP_STD))
EFF = ([0.5,0.5,0.5],[0.5,0.5,0.5])
SPEC = {
 "dinov2":   dict(bb="vit_large_patch14_dinov2.lvd142m",     norm=IMN, kind="timm"),
 "convnext": dict(bb="convnextv2_large.fcmae_ft_in22k_in1k", norm=IMN, kind="timm"),
 "faceptor": dict(bb=None,                                    norm=CLP, kind="faceptor"),
}


class TestDS(Dataset):
    def __init__(self, files, norm):
        self.files = files
        self.t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                     transforms.Normalize(*norm)])
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.t(Image.open(CROPS/self.files[i]).convert("RGB")), i


@torch.no_grad()
def predict(model, files, norm, dev):
    dl = DataLoader(TestDS(files, norm), batch_size=256, num_workers=8, pin_memory=True)
    out = np.zeros(len(files), np.float32); model.eval()
    for x, idx in dl:
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = (model(x).float() + model(torch.flip(x,[3])).float()) / 2   # TTA miroir
        out[idx.numpy()] = p.cpu().numpy()
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    files = pd.read_csv("data/test_students.csv")["filename"].tolist()
    Path(a.out).mkdir(exist_ok=True)
    for arch, c in SPEC.items():
        for k in range(5):
            ck = torch.load(f"{a.weights_dir}/ckpt_{arch}_f{k}.pt", map_location=dev, weights_only=False)
            model = (T.RegressionModel(c["bb"], pretrained=False) if c["kind"]=="timm"
                     else FaceptorRegressor(pretrained=False)).to(dev)
            model.load_state_dict(ck["model_state_dict"])
            tp = predict(model, files, c["norm"], dev)
            pd.DataFrame({"filename": files, "pred": tp}).to_csv(f"{a.out}/testpred_{arch}_f{k}.csv", index=False)
            print(f"  {arch} f{k} -> testpred  (mean={tp.mean():.4f})", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
