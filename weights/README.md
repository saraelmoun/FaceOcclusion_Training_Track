# Poids

Les **3 poids fold-0** (DINOv2, ConvNeXt, Faceptor) sont disponibles, trop volumineux
pour git → à héberger sur Hugging Face Hub / Git LFS / cloud.

| Fichier | Taille |
|---|---|
| dinov2_fold0_best.pt | ~1.2 Go |
| convnextv2_large_myfold_best.pt | ~0.8 Go |
| faceptor_b16_best.pt | ~0.35 Go |

Les 12 poids fold 1-4 n'ont pas été retenus (voir note de reproductibilité du README
racine). La soumission se reproduit sans eux via `predictions/` + `assemble.py`.
