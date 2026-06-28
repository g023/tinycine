#!/usr/bin/env bash
# =============================================================================
# download_models.sh — fetch the LTX-2.3 weights tinycine needs into ./models/
# =============================================================================
# tinycine ships NO model weights (they are © Lightricks). This script pulls
# them from the Hugging Face Hub. It is idempotent and resumable — re-run it any
# time; finished files are skipped and partial downloads continue.
#
#   ./download_models.sh             # download everything that is missing
#   ./download_models.sh --no-upscalers   # skip the optional 2-stage upscalers
#
# What it fetches:
#   1. diffusers/LTX-2.3-Distilled-Diffusers  (~95 GB) -> models/diffusers-distilled/
#        the diffusers-format distilled repo: transformer + VAEs + Gemma-3-12B
#        text encoder + tokenizer. This is the only thing tinycine needs to run.
#   2. Lightricks/LTX-2.3 upscalers           (~1.3 GB) -> models/upscalers/
#        optional; only used by `--upscale`. Skip with --no-upscalers.
#
# Requirements: the Hugging Face CLI (`pip install -U 'huggingface_hub[cli]'`)
#               and ~100 GB free disk. The big repo is the LTX-2 Community License.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$ROOT/models"
LOGDIR="$ROOT/logs"
mkdir -p "$MODELS/upscalers" "$LOGDIR"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1

WANT_UPSCALERS=1
case "${1:-}" in
  --no-upscalers) WANT_UPSCALERS=0 ;;
  -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "unknown arg: $1 (use --no-upscalers or no arg)"; exit 2 ;;
esac

ts()  { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; }

if ! command -v hf >/dev/null 2>&1 && ! python3 -c 'import huggingface_hub' 2>/dev/null; then
  say "ERROR: huggingface_hub is not installed."
  say "       Run:  pip install -U 'huggingface_hub[cli]' hf_transfer"
  exit 1
fi

say "===== tinycine model download ====="
say "target: $MODELS   (free disk: $(df -h "$ROOT" | awk 'NR==2{print $4}'))"

# --- 1) the main diffusers-distilled repo (with live progress) -----------------
say ">>> diffusers/LTX-2.3-Distilled-Diffusers  (~95 GB) — this is the long one"
if python3 "$ROOT/scripts/fetch_diffusers_repo.py" 2>&1 | tee -a "$LOGDIR/download.log"; then
  say "OK   diffusers-distilled repo present"
else
  say "FAIL diffusers-distilled repo — re-run ./download_models.sh to resume"
  exit 1
fi

# --- 2) optional upscalers -----------------------------------------------------
if [ "$WANT_UPSCALERS" = 1 ]; then
  for f in ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
           ltx-2.3-temporal-upscaler-x2-1.0.safetensors; do
    if [ -f "$MODELS/upscalers/$f" ]; then
      say "OK   upscalers/$f (present)"
      continue
    fi
    say ">>> Lightricks/LTX-2.3 :: $f"
    if hf download Lightricks/LTX-2.3 "$f" --local-dir "$MODELS/upscalers" \
         >>"$LOGDIR/download.log" 2>&1; then
      say "DONE upscalers/$f"
    else
      say "WARN could not fetch upscalers/$f (optional — only needed for --upscale)"
    fi
  done
  # flatten any nested layout hf may create
  find "$MODELS/upscalers" -mindepth 2 -name '*.safetensors' \
    -exec mv -f {} "$MODELS/upscalers/" \; 2>/dev/null || true
else
  say "skipping upscalers (--no-upscalers)"
fi

# tidy HF caches (metadata only; weights kept)
find "$MODELS" -type d -name '.cache' -prune -exec rm -rf {} + 2>/dev/null || true

say "----------------------------------------------------------------"
say "total models size: $(du -sh "$MODELS" 2>/dev/null | cut -f1)"
say "===== models ready — now run ./runme.sh ====="
