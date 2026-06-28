# 🛠 TinyCine — Setup Guide

Everything you need to get tinycine generating video on your machine, from a clean OS to your
first clip. If you just want the three commands, see the [Quick Start](README.md#-quick-start);
this file is the complete walkthrough plus troubleshooting.

---

## 1. Hardware & OS requirements

| | Minimum / reference |
|---|---|
| **GPU** | NVIDIA **Ampere or newer (sm_86+)**, **12 GB** VRAM. Reference card: **RTX 3060 12 GB**. (A 4070/3080/4090 etc. also works and is faster.) |
| **System RAM** | **~62 GB.** This is not optional — CPU offload streams the model from system RAM, so a 16–32 GB box will thrash or fail. |
| **Disk** | **~100 GB free** for the model repo (~95 GB) plus a little headroom. |
| **OS** | Linux (tested on Ubuntu-class, kernel 6.x). |
| **Driver** | Recent NVIDIA driver (580.x or newer recommended) with **CUDA 12.x** runtime support. |
| **Python** | 3.10 – 3.12. |
| **ffmpeg/PyAV** | Provided via the `av` pip wheel — no system ffmpeg needed. |

Check your GPU and driver:

```bash
nvidia-smi            # should show your GPU, 12288 MiB, and "CUDA Version: 12.x"
python3 --version     # 3.10–3.12
df -h .               # confirm ~100 GB free where you'll put models/
free -g               # confirm ~60 GB total RAM
```

> **Why so much system RAM?** The two largest components (the ~22B transformer and the
> Gemma-3-12B text encoder) are too big for 12 GB of VRAM, so they live in system RAM and are
> streamed onto the GPU a piece at a time. 62 GB comfortably holds the offloaded weights with
> room to spare. With far less RAM, the OS will swap and runs become painfully slow or crash.

---

## 2. Install the runtime

From the `tinycine/` directory:

```bash
# 2a. Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2b. Install PyTorch FIRST, from the CUDA 12.x wheel index.
#     (Pick the index matching your CUDA — cu124 works on CUDA 12.1–12.9 drivers.)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 2c. Install the rest of the dependencies
pip install -r requirements.txt
```

> **Order matters.** Installing `torch` from the CUDA wheel index *before* the other packages
> ensures pip doesn't pull a CPU-only build. If you ever see "CUDA not available", this is the
> first thing to recheck (see Troubleshooting).

Verify the GPU is visible to PyTorch:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect e.g.:  2.6.0+cu124 True NVIDIA GeForce RTX 3060
```

---

## 3. Download the model weights

tinycine ships **no weights** — they're ~95 GB and © Lightricks. Fetch them with:

```bash
./download_models.sh
```

This downloads (resumably — safe to Ctrl-C and re-run):

1. **`diffusers/LTX-2.3-Distilled-Diffusers`** (~95 GB) → `models/diffusers-distilled/`
   — the transformer, both VAEs, the Gemma-3-12B text encoder, and the tokenizer. This is the
   only thing required to generate video.
2. **Two LTX-2.3 latent upscalers** (~1.3 GB) → `models/upscalers/`
   — optional, only used by `--upscale`. Skip them with `./download_models.sh --no-upscalers`.

Progress is printed live (GB pulled, MB/s) and also logged to `logs/download.log`. The big repo
takes a while — it's network-bound, so time depends on your connection.

> **Hugging Face access.** The repos are public, but if you hit a rate limit or auth prompt, run
> `hf auth login` once with a free [HF token](https://huggingface.co/settings/tokens).
> `hf_transfer` (in `requirements.txt`) is enabled automatically for faster downloads.

When it finishes you should have:

```
models/
├── diffusers-distilled/   (~95 GB — transformer, vae, text_encoder, tokenizer, ...)
└── upscalers/             (~1.3 GB — spatial + temporal, optional)
```

---

## 4. Generate your first clip

```bash
./runme.sh
```

That renders the default prompt at the **balanced** preset (768×768, 49 frames, with audio).
To use your own prompt, pass it as an argument:

```bash
./runme.sh "a paper boat sailing down a rain-soaked gutter, macro, shallow depth of field"
```

The output lands at `assets/outputs/tinycine.mp4` — an H.264 MP4 with a synchronized AAC audio
track (LTX-2.3 generates picture and sound jointly).

### What you'll see in the log

The CLI narrates exactly what it's doing and **measures** memory as it goes:

```
[ltxv] Preflight: MemAvailable 62.0 GB, VRAM total 12.0 GB on NVIDIA GeForce RTX 3060
[ltxv] Phase A: loading Gemma text encoder (4bit=True) ...
[ltxv] Phase A done; Gemma freed. host MemAvailable now 58.x GB
[ltxv] Phase B: lever engaged -> transformer fp8 layerwise-casting ...
[ltxv] Phase B: lever engaged -> vae.enable_tiling()
[ltxv] Phase B: lever engaged -> enable_sequential_cpu_offload()
[ltxv]   step 1/8  dt 9.x s  VRAM 2.5 GB
   ...
[ltxv] MEASURED: wall 205.x s  VRAM peak alloc 2.52 GB / reserved 7.63 GB ...
[ltxv] WROTE assets/outputs/tinycine.mp4
```

If any memory shortcut is taken to fit, it is **logged, never silent**.

---

## 5. Tuning & flags

`runme.sh` exposes the common knobs at the top of the file (prompt, quality, seed, upscale).
For the full set, call the CLI directly:

```bash
python3 scripts/ltxv_cli.py --help
```

Most-used flags:

| flag | meaning | default |
|------|---------|---------|
| `--prompt` | text prompt (required) | — |
| `--out` | output `.mp4` path (required) | — |
| `--quality` | preset: `fast` / `balanced` / `high` / `none` | `balanced` |
| `--size` | `WxH`, each divisible by 32 (**overrides** preset) | preset |
| `--frames` | frame count, must be `8k+1` (e.g. 25, 49, 73) | preset |
| `--cfg` | guidance scale (distilled best 1.0–2.0; `>1` enables negative prompt) | preset |
| `--seed` | RNG seed (set for reproducibility) | `42` |
| `--crf` | libx264 quality (lower = better/bigger; 18 ≈ near-lossless) | `18` |
| `--upscale` | `none` / `spatial` / `temporal` / `spatial+temporal` | `none` |
| `--keep-base` | when upscaling, also keep the low-res clip as `<out>.base.mp4` | off |
| `--offload` | `sequential` (lowest VRAM) / `model` / `none` | `sequential` |
| `--no-fp8` | disable transformer fp8 layerwise casting | off |
| `--no-vae-tiling` | disable tiled VAE decode (uses more VRAM) | off |
| `--te-bf16` | load Gemma in bf16 instead of 4-bit (needs ~50 GB extra RAM) | off |
| `--min-ram-gb` | refuse to start below this much free system RAM | `8.0` |

**Hard constraints** (the CLI errors loudly rather than fixing silently):
- Width and height must each be **divisible by 32**.
- Frame count must be **`8k+1`** (25, 49, 73, 97, …).

---

## 6. Troubleshooting

**`CUDA not available` / runs on CPU.**
You likely installed a CPU-only torch. Reinstall from the CUDA index *first*:
```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python3 -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**Out of VRAM (`CUDA out of memory`).**
This shouldn't happen at the default settings (peak ~2.5 GB). If you changed flags: keep
`--offload sequential`, keep fp8 and VAE tiling on, and drop `--size`/`--frames`. Make sure no
other process is using the GPU (`nvidia-smi`).

**The box gets sluggish / a run gets OOM-killed by the system.**
That's *system* RAM pressure, not VRAM. Confirm you have ~62 GB and that `--te-bf16` is **off**
(bf16 Gemma needs ~50 GB extra). The CLI runs a host-RAM preflight and refuses to start below
`--min-ram-gb`. If a CUDA process was killed mid-run and free RAM never recovers, a reboot clears
any leaked locked pages.

**`bitsandbytes` import / 4-bit errors.**
Ensure `bitsandbytes>=0.44` installed against your CUDA build. Reinstall inside the venv:
`pip install -U bitsandbytes`.

**No audio in the output / `av` errors.**
Audio is muxed via PyAV; make sure `av` installed cleanly (`pip install -U av`). The model emits
audio jointly — no flag is needed.

**Download stalls or fails partway.**
`./download_models.sh` is resumable — just run it again; finished files are skipped. Check
`logs/download.log`. For auth/rate limits, run `hf auth login`.

**Wrong frame count or size error.**
Honor the constraints: W/H divisible by 32, frames = `8k+1`.

---

## 7. Quick reference

```bash
# one-time setup
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt

# get the weights (resumable)
./download_models.sh

# generate
./runme.sh "your prompt here"

# advanced
python3 scripts/ltxv_cli.py --help
```

Have fun. 🎬
