"""Driver générique : entraîne 1 architecture sur 1 fold (val = fold k, train = reste).
Sauve les prédictions OOF (fold k) et les prédictions test (avec TTA miroir).

Usage : python run_fold.py --arch {dinov2,convnext,faceptor} --fold k --device cuda:N
"""
import argparse, os, sys, numpy as np, pandas as pd, torch
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import train_dinov2_large as T
from run_faceptor_e2e import FaceptorRegressor, CLIP_MEAN, CLIP_STD

CROPS = os.environ.get("CROPS_DIR","/root/crops/Crop_224_5fp_100K").rstrip("/")+"/"
IMN = ([0.485,0.456,0.406],[0.229,0.224,0.225]); CLP = (list(CLIP_MEAN),list(CLIP_STD))
EFF = ([0.5,0.5,0.5],[0.5,0.5,0.5])     # normalisation native EfficientNetV2 (tf_)
ARCH = {
 "dinov2":       dict(bb="vit_large_patch14_dinov2.lvd142m",     norm=IMN, lr=(5e-6,5e-5), kind="timm"),
 "convnext":     dict(bb="convnextv2_large.fcmae_ft_in22k_in1k", norm=IMN, lr=(2e-5,2e-4), kind="timm"),
 "faceptor":     dict(bb=None,                                   norm=CLP, lr=(5e-6,5e-5), kind="faceptor"),
 "efficientnet": dict(bb="tf_efficientnetv2_l.in21k_ft_in1k",    norm=EFF, lr=(2e-5,2e-4), kind="timm"),
}

def make_model(arch):
    c = ARCH[arch]
    return T.RegressionModel(c["bb"], pretrained=True) if c["kind"]=="timm" else FaceptorRegressor()

def tf(norm, train):
    mean, std = norm
    if train:
        return transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.10,0.10,0.05,0)],0.5),
            transforms.RandomRotation(3), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

class TestDS(Dataset):
    def __init__(self, files, norm):
        self.files=files; self.t=transforms.Compose([transforms.Resize((224,224)),
            transforms.ToTensor(), transforms.Normalize(*norm)])
    def __len__(self): return len(self.files)
    def __getitem__(self,i): return self.t(Image.open(CROPS+self.files[i]).convert("RGB")), i

@torch.no_grad()
def predict_test(model, files, norm, dev):
    dl=DataLoader(TestDS(files,norm),batch_size=256,shuffle=False,num_workers=16,pin_memory=True)
    out=np.zeros(len(files),np.float32); model.eval()
    for x,idx in dl:
        x=x.to(dev,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=(model(x).float()+model(torch.flip(x,[3])).float())/2          # TTA miroir
        out[idx.numpy()]=p.cpu().numpy()
    return np.clip(out,0,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--arch",required=True,choices=list(ARCH))
    ap.add_argument("--fold",type=int,required=True); ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--outdir",default=str(T.RUN_DIR))
    a=ap.parse_args(); dev=torch.device(a.device); c=ARCH[a.arch]
    from pathlib import Path as _P
    OUT=_P(a.outdir); OUT.mkdir(parents=True,exist_ok=True)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False  # déterminisme best-effort
    oof_path=OUT/f"oof_{a.arch}_f{a.fold}.csv"
    if oof_path.exists(): print(f"[{a.arch} f{a.fold}] déjà fait — skip",flush=True); return

    df=pd.read_csv(T.OUT_PATH)
    df_tr=df[df.fold!=a.fold].reset_index(drop=True); df_va=df[df.fold==a.fold].reset_index(drop=True)
    T.seed_everything(T.SEED)
    tr_ds=T.OcclusionDataset(df_tr,T.IMAGE_DIR,transform=tf(c["norm"],True),training=True,return_meta=True)
    va_ds=T.OcclusionDataset(df_va,T.IMAGE_DIR,transform=tf(c["norm"],False),training=True,return_meta=True)
    bs=T.RobustOccGenderBatchSampler(df=df_tr,batch_size=128,bin_col="occ_bin_robust",gender_col="gender",
        bin_quotas={"low_[0,0.1)":64,"medium_[0.1,0.3)":50,"high_[0.3,1.01)":14},tau=0.75,
        n_batches=len(df_tr)//128,seed=T.SEED)
    tl=DataLoader(tr_ds,batch_sampler=bs,num_workers=8,pin_memory=True)
    vl=DataLoader(va_ds,batch_size=128,shuffle=False,num_workers=8,pin_memory=True)

    model=make_model(a.arch).to(dev)
    opt=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":c["lr"][0]},
                           {"params":model.head.parameters(),"lr":c["lr"][1]}],weight_decay=1e-4)
    nsteps=12*len(tl); sched=T.get_cosine_schedule_with_warmup(opt,int(0.05*nsteps),nsteps,0.05)
    best=(np.inf,None,None)
    for ep in range(1,13):
        T.train_one_epoch_amp(model,tl,opt,dev,sched)
        vp=T.predict_loader(model,vl,dev)
        s=T.evaluate_predictions(vp,model_name=f"{a.arch}_f{a.fold}_ep{ep}",verbose=False)["val_score_global"]
        print(f"[{a.arch} f{a.fold}] ep{ep:02d} score={s:.6f}",flush=True)
        if np.isfinite(s) and s<best[0]:
            best=(s,vp.copy(),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[2])
    # >>> SAUVEGARDE DU CHECKPOINT (le correctif : on garde les poids cette fois) <<<
    torch.save({"model_state_dict":best[2],"arch":a.arch,"fold":a.fold,
                "best_score":float(best[0])}, OUT/f"ckpt_{a.arch}_f{a.fold}.pt")
    best[1].to_csv(oof_path,index=False)                                     # OOF val (fold k)
    test=pd.read_csv(_P(__file__).resolve().parent.parent/"data"/"test_students.csv"); files=test.filename.tolist()
    tp=predict_test(model,files,c["norm"],dev)
    pd.DataFrame({"filename":files,"pred":tp}).to_csv(OUT/f"testpred_{a.arch}_f{a.fold}.csv",index=False)
    print(f"[{a.arch} f{a.fold}] DONE best={best[0]:.6f}  ckpt sauvé",flush=True)

if __name__ == "__main__":
    main()
