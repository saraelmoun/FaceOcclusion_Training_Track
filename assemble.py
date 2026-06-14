"""Assemblage déterministe : prédictions par fold → soumission finale.

Reproduit EXACTEMENT le fichier soumis, sans GPU ni poids, à partir des seuls
CSV de prédictions (predictions/). Étapes :
  1. Bagging : pour chaque architecture, moyenne des 5 modèles-folds (test).
  2. Ensemble : moyenne des 3 architectures (DINOv2, ConvNeXt, Faceptor).
  3. Calibration isotone pondérée, ajustée sur l'OOF complet (100k), appliquée au test.

Produit 3 fichiers de soumission :
  - test_predictions_v2_5fold.csv            (sans calibration)
  - test_predictions_v2_5fold_calib_sq.csv   (calibration (1/30+y)^2)
  - test_predictions_v2_5fold_calib_cube.csv (calibration (1/30+y)^3)

Usage : python assemble.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predictions"
OUT = ROOT / "submission"
ARCHS = ["dinov2", "convnext", "faceptor"]
FOLDS = [0, 1, 2, 3, 4]


def load_oof():
    """OOF complet 100k : chaque image prédite par le modèle qui ne l'a pas vue."""
    cols = ["filename", "FaceOcclusion", "pred"]
    oof = {}
    for a in ARCHS:
        parts = [pd.read_csv(PRED / f"oof_{a}_f{k}.csv")[cols] for k in FOLDS]
        oof[a] = pd.concat(parts, ignore_index=True).rename(columns={"pred": a}).set_index("filename")
    M = oof[ARCHS[0]][["FaceOcclusion"]].copy()
    for a in ARCHS:
        M[a] = oof[a][a]
    return M.dropna()


def bag_test():
    """Bagging test : moyenne des 5 folds par archi, puis des 3 archis."""
    test = pd.read_csv(ROOT / "data/test_students.csv")
    bag = {a: np.mean([pd.read_csv(PRED / f"testpred_{a}_f{k}.csv")["pred"].values
                       for k in FOLDS], axis=0) for a in ARCHS}
    ens = np.mean([bag[a] for a in ARCHS], axis=0).clip(0, 1)
    return test["filename"].values, ens


def make_submission(filenames, pred, name):
    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame({"filename": filenames, "FaceOcclusion": np.clip(pred, 0, 1), "gender": "x"})
    df.to_csv(OUT / name, index=False)
    print(f"  écrit {name:42s} mean={df.FaceOcclusion.mean():.4f} max={df.FaceOcclusion.max():.4f}")


def main():
    M = load_oof()
    yv = M["FaceOcclusion"].values
    ens_oof = M[ARCHS].mean(axis=1).values.clip(0, 1)
    files, ens_test = bag_test()
    print(f"OOF {len(M)} images | test {len(files)} images")

    make_submission(files, ens_test, "test_predictions_v2_5fold.csv")
    for k, tag in [(2, "sq"), (3, "cube")]:
        ir = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(
            ens_oof, yv, sample_weight=(1 / 30 + yv) ** k)
        make_submission(files, ir.predict(ens_test), f"test_predictions_v2_5fold_calib_{tag}.csv")


if __name__ == "__main__":
    main()
