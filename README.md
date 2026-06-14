<div align="center">

# 👁️ FacePredict — Prédiction d'occlusion du visage

**Estimer, pour chaque visage, la fraction occultée `FaceOcclusion ∈ [0, 1]`**

*Group 11 · Data Challenge IMT 2026*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Score](https://img.shields.io/badge/Leaderboard-0.00098-success)
![Reproducible](https://img.shields.io/badge/Reproductible-✅-blue)

</div>

---

## Démarrage rapide

```bash
git clone https://github.com/saraelmoun/facepredict-occlusion-v2 && cd facepredict-occlusion-v2
pip install -r requirements.txt
bash weights/download_weights.sh          # récupère les 15 modèles (HF)
export CROPS_DIR=/chemin/vers/crops_224   # tes images
python inference.py --weights_dir weights && python assemble.py   # → submission/
```

> 📁 Le résultat atterrit dans `submission/`. **Aucun entraînement requis** pour reproduire le score.

---

## 🎯 La problématique

On dispose de **crops de visages 224×224**. Pour chacun, il faut prédire **quelle proportion du
visage est occultée** (cheveux, lunettes, main, chapeau…), un nombre entre **0** (visage net) et
**~0.5** (très occulté).

La difficulté vient de la **métrique d'évaluation**, qui n'est pas une simple erreur moyenne :

```
  w = 1/30 + y                          ← chaque visage est pondéré par son occlusion
  Err_g = Σ w·(p − y)² / Σ w            ← erreur pondérée, calculée PAR GENRE g
  Score = (Err_F + Err_M)/2  +  |Err_F − Err_M|
                  ▲                          ▲
            erreur moyenne            pénalité d'inéquité entre genres
```

Deux conséquences qui dirigent toute la solution :
- 🔸 **les fortes occlusions comptent ~16× plus** que les visages nets (poids `w`) ;
- 🔸 **un modèle inéquitable entre genres est puni** (terme `|Err_F − Err_M|`).

---

## 💡 La solution

Un **ensemble de 3 réseaux complémentaires**, chacun fine-tuné sur 5 découpages, consolidé par
bagging puis une calibration finale :

```
                  ┌─────────────────────────────────────────────┐
   image 224×224  │   3 backbones fine-tunés (×5 folds = 15)     │
        │         │                                             │
        ├────────►│  🧠 DINOv2-L     (ViT, attention globale)   │──┐
        ├────────►│  🔲 ConvNeXt V2-L (conv, détail local)       │──┤
        └────────►│  🙂 Faceptor      (CLIP ViT-B/16, visage)    │──┤
                  └─────────────────────────────────────────────┘  │
                                                                    ▼
              bagging 5-fold ──► moyenne des 3 archis ──► calibration isotone
                                                                    │
                                                                    ▼
                                               📄 FaceOcclusion ∈ [0,1]
```

| Étage | Rôle | Pourquoi |
|---|---|---|
| **3 architectures** | DINOv2 (attention) + ConvNeXt (convolution) + Faceptor (spécialisé visage) | inductive biases complémentaires → erreurs décorrélées |
| **Échantillonnage robuste** | sur-représente les fortes occlusions dans chaque batch | la métrique les survalorise |
| **Loss pondérée** `w=1/30+y` | alignée sur la métrique | optimise ce qui est noté |
| **Bagging 5-fold** | 15 modèles, moyennés | réduction de variance |
| **Calibration isotone** | corrige le biais résiduel | gain sur la distribution test |

---

## 🚀 Utilisation — deux parcours

### 🅰️ Reproduire le résultat (inférence seule) — recommandé

> Tu as les **crops** et tu veux **la soumission**, sans rien entraîner. ~10 min.

```bash
pip install -r requirements.txt
bash weights/download_weights.sh                 # 15 poids (~12 Go) depuis Hugging Face
export CROPS_DIR=/chemin/vers/crops_224
python inference.py --weights_dir weights        # poids → prédictions (avec TTA)
python assemble.py                               # → submission/test_predictions_v2_5fold*.csv
```

✅ **Ne nécessite que** : ce dépôt + les crops + les poids (HF). Pas de GPU d'entraînement, pas
de checkpoint externe.

### 🅱️ Tout ré-entraîner depuis zéro

> Tu veux **régénérer les 15 modèles**. ~5 h GPU. Le split (`data/train_with_folds.csv`, seed 42)
> est fourni — rien n'est figé en dur, tout passe par des variables d'environnement.

```bash
export CROPS_DIR=/chemin/vers/crops_224
# checkpoint Faceptor officiel (public) — requis seulement pour entraîner Faceptor :
huggingface-cli download saraelmoun/facepredict-icl-extractor-weights \
    faceptor_checkpoint_rank0_iter_50000.pth.tar --local-dir weights
export FACEPTOR_CKPT=weights/faceptor_checkpoint_rank0_iter_50000.pth.tar

for arch in dinov2 convnext faceptor; do
  for k in 0 1 2 3 4; do
    python src/run_fold.py --arch $arch --fold $k --device cuda:0 --outdir weights
  done
done
python inference.py --weights_dir weights && python assemble.py
```

✅ **Nécessite** : ce dépôt + crops + checkpoint Faceptor (public, HF). DINOv2/ConvNeXt sont
auto-téléchargés par `timm`.
⚠️ Un ré-entraînement est un **nouveau tirage** (GPU non bit-déterministe) → score *équivalent*,
pas identique. Pour le résultat exact, utiliser les poids fournis (parcours 🅰️).

| | Parcours 🅰️ Inférence | Parcours 🅱️ Ré-entraînement |
|---|---|---|
| Temps | ~10 min | ~5 h GPU |
| Besoin | crops + poids (HF) | crops + checkpoint Faceptor (HF) |
| Résultat | **soumission exacte** | modèles équivalents (nouveau tirage) |

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| 🏆 Score leaderboard | **0.00098** |
| Estimation robuste (OOF, 100k images) | métrique 0.000839 |
| Configuration | ensemble 3 archis × bagging 5-fold + TTA + calibration |

---

## 📦 Ressources (Hugging Face)

| Repo HF | Contenu | Usage |
|---|---|---|
| [`facepredict-occlusion-v2-weights`](https://huggingface.co/saraelmoun/facepredict-occlusion-v2-weights) | 15 checkpoints (~12 Go) | inférence (`download_weights.sh`) |
| [`facepredict-icl-extractor-weights`](https://huggingface.co/saraelmoun/facepredict-icl-extractor-weights) | checkpoint Faceptor officiel | ré-entraînement Faceptor |

---

## 📂 Arborescence

```
📁 facepredict-occlusion-v2/
├── 📄 README.md
├── 📄 requirements.txt          versions épinglées
├── 📁 src/                      cœur du code
│   ├── train_dinov2_large.py    split · dataset · sampler · métrique · boucle d'entraînement
│   ├── run_fold.py              entraîne 1 archi × 1 fold → checkpoint
│   ├── run_faceptor_e2e.py      backbone Faceptor
│   └── clip_vit.py              CLIP ViT vendorisé (zéro dépendance externe)
├── 📄 inference.py              poids → prédictions test (TTA miroir)
├── 📄 assemble.py               bagging + ensemble + calibration → soumission
├── 📁 data/                     train_with_folds.csv (split seed 42) · test_students.csv
├── 📁 predictions/              30 CSV (OOF + test/fold) → repro sans GPU
├── 📁 weights/                  download_weights.sh (poids sur HF)
└── 📁 submission/               fichiers de soumission
```

---

## ⚙️ Détails techniques

<details>
<summary><b>Les 3 backbones en profondeur</b></summary>

| Modèle | Type | Params | Pré-entraînement | Représentation |
|---|---|---|---|---|
| **DINOv2-L** (`vit_large_patch14_dinov2`) | ViT, 24 blocs, 1024-d, 16 têtes | 304 M | auto-supervisé (self-distillation, LVD-142M) | CLS 1024-d |
| **ConvNeXt V2-L** (`convnextv2_large.fcmae_ft_in22k_in1k`) | conv hiérarchique + GRN | 198 M | FCMAE auto-sup. → IN-22k/1k | avg-pool 1536-d |
| **Faceptor** (CLIP ViT-B/16, `lxq1000/Faceptor`) | ViT, 12 blocs, 768-d | ~86 M | CLIP → fine-tuné 6 tâches visage | CLS projeté 512-d |

Tête commune : `LayerNorm → Linear(·,512) → GELU → Dropout → Linear(512,1)`.
Loss : `weighted_mse` avec `w = 1/30 + y`. Sampler : quotas par bin d'occlusion × genre.

</details>

<details>
<summary><b>Reproductibilité — garanties et limites</b></summary>

- **Reproduction depuis les prédictions** (`assemble.py`) : **bit-exacte**, sans GPU.
- **Reproduction depuis les poids** (`inference.py`) : à ~1e-3 près (bruit bf16) ; aucune
  dépendance externe (Faceptor construit sans son checkpoint officiel à l'inférence).
- **Tous les chemins sont paramétrables** (`CROPS_DIR`, `FACEPTOR_CKPT`, `OUTDIR`, `FOLDS_CSV`) —
  rien n'est figé en dur.
- Le split est groupé par **identité** (aucune fuite train/val) et stratifié genre × occlusion.

</details>

<div align="center">

---

*Reproductible de bout en bout · poids et features hébergés sur Hugging Face*

</div>
