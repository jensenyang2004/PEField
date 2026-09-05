#!/usr/bin/env bash
# Download the exact checkpoints required by PE-Field inference.
#
# Usage (run from anywhere):
#   HF_TOKEN=hf_... bash scripts/download_model_weights.sh
#
# Before running, accept the access conditions for FLUX.1-Kontext-dev at:
# https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
# Alternatively, authenticate once with: hf auth login

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v hf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
The Hugging Face CLI (`hf`) is required. Install it in the target environment:
  pip install -U "huggingface_hub[cli]"
Then authenticate with `hf auth login` (after accepting the FLUX license), or
set HF_TOKEN to a Hugging Face token with access.
EOF
    exit 1
fi

download() {
    # `HF_TOKEN`, when supplied, is consumed by the Hugging Face CLI.  Without
    # it, the CLI uses the token saved by `hf auth login`.
    hf download "$@"
}

echo 'Downloading the official FLUX.1-Kontext-dev pipeline (without transformer)...'
# The root flux1-kontext-dev.safetensors is a complete single-file checkpoint
# and would duplicate the PE-Field transformer.  The allow-list below contains
# only the components FluxKontextPipeline.from_pretrained() needs when a custom
# transformer is passed by infer_viewchanger_single_v2.py.
download black-forest-labs/FLUX.1-Kontext-dev \
    --include \
        'model_index.json' \
        'scheduler/*' \
        'text_encoder/*' \
        'text_encoder_2/*' \
        'tokenizer/*' \
        'tokenizer_2/*' \
        'vae/*' \
    --local-dir "$ROOT_DIR/FLUX.1-Kontext-dev"

echo 'Downloading PE-Field transformer weights...'
# Hugging Face preserves repository paths under --local-dir. Download each file
# into a staging directory and immediately move it to the path expected by the
# inference script. This avoids retaining a second ~24 GB transformer copy.
STAGING_DIR="$ROOT_DIR/.model-download-staging"
mkdir -p "$STAGING_DIR" "$ROOT_DIR/checkpoints/transformer"

download_transformer_file() {
    local filename="$1"
    download yunpeng1998/PE-Field "$filename" --local-dir "$STAGING_DIR"
    mv -f "$STAGING_DIR/$filename" \
        "$ROOT_DIR/checkpoints/transformer/$(basename "$filename")"
}

download_transformer_file 'FLUX.1-Kontext-dev/transformer/config.json'
download_transformer_file 'FLUX.1-Kontext-dev/transformer/diffusion_pytorch_model.safetensors.index.json'
download_transformer_file 'FLUX.1-Kontext-dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors'
download_transformer_file 'FLUX.1-Kontext-dev/transformer/diffusion_pytorch_model-00002-of-00003.safetensors'
download_transformer_file 'FLUX.1-Kontext-dev/transformer/diffusion_pytorch_model-00003-of-00003.safetensors'

echo 'Downloading MoGe 2 ViT-L normal weights...'
mkdir -p "$ROOT_DIR/moge-2-vitl-normal"
download Ruicheng/moge-2-vitl-normal model.pt \
    --local-dir "$ROOT_DIR/moge-2-vitl-normal"

cat <<'EOF'

Done. The expected paths are now:
  FLUX.1-Kontext-dev/             (official pipeline, no stock transformer)
  checkpoints/transformer/        (PE-Field transformer)
  moge-2-vitl-normal/model.pt     (MoGe)
EOF
