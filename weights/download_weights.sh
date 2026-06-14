#!/bin/bash
# Télécharge les 15 checkpoints depuis Hugging Face Hub.
# Repo public : aucun login requis (téléchargement direct).
HF_REPO="${1:-saraelmoun/facepredict-occlusion-v2-weights}"
pip install -q huggingface_hub
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="$HF_REPO", repo_type="model", local_dir=".",
                      allow_patterns=["ckpt_*.pt"])
print("15 checkpoints téléchargés dans", p)
PY
