#!/usr/bin/env bash
# =============================================================================
# runme.sh — generate an audio+video clip with tinycine on a 12 GB RTX 3060.
# =============================================================================
# This is the one-command runner. It uses the settings TUNED to perform best on
# the RTX 3060 12 GB: the distilled 8-step schedule, fp8 layerwise casting,
# sequential CPU offload, 4-bit Gemma, and tiled VAE decode — measured well under
# the 12 GB ceiling. Just:
#
#     ./runme.sh                       # render the default prompt at "balanced"
#     ./runme.sh "your prompt here"    # render your own prompt
#
# Edit the knobs below to change prompt/quality/output. For every flag, see:
#     python3 scripts/ltxv_cli.py --help
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── EDIT THESE ───────────────────────────────────────────────────────────────
# A prompt passed on the command line ($1) wins over this default.
PROMPT="${1:-A red fox trotting through a snowy forest at dawn, cinematic, soft golden light}"
OUT="assets/outputs/tinycine.mp4"

# QUALITY preset — the performance/detail sweet spots measured on the 3060:
#   fast      512x512x25,  cfg 1.0  → ~105 s   (drafts / fast iteration)
#   balanced  768x768x49,  cfg 1.5  → ~3.4 min (the everyday clip — DEFAULT)
#   high      1024x1024x49, cfg 2.0 → ~4.8 min (hero shots)
# All three stay ~5x under the 12 GB VRAM ceiling.
QUALITY="balanced"

SEED=42
NEGATIVE=""                 # empty = the model's default negative prompt
UPSCALE="none"              # none | spatial | temporal | spatial+temporal (optional 2nd stage)
# ─────────────────────────────────────────────────────────────────────────────

# Activate the CUDA venv if present (see SETUP.md).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "NOTE: .venv not found — see SETUP.md to create it (python3 -m venv .venv ...)." >&2
fi

mkdir -p "$(dirname "$OUT")"

ARGS=(
  --prompt   "$PROMPT"
  --out      "$OUT"
  --quality  "$QUALITY"
  --seed     "$SEED"
  --upscale  "$UPSCALE"
)
[[ -n "$NEGATIVE" ]] && ARGS+=( --negative "$NEGATIVE" )

echo "tinycine → generating \"$PROMPT\""
echo "          quality=$QUALITY  out=$OUT"
python3 scripts/ltxv_cli.py "${ARGS[@]}"
echo
echo "Done. Your clip (with audio) is at: $OUT"
