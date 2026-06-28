<div align="center">

# 🎬 TinyCine

### Datacenter-grade AI video, generated on the GPU you already own.

**Run Lightricks' [LTX-2.3](https://github.com/Lightricks/LTX-Video) — a ~22-billion-parameter audio+video diffusion model — on a single 12 GB RTX 3060.**

*Text in. Cinematic clip + synced audio out. No cloud, no rental GPU, no ComfyUI.*

[Quick Start](#-quick-start) · [How it fits](#-how-on-earth-does-this-fit) · [Presets](#-pick-your-preset) · [Setup](SETUP.md) · [License](#-license--credits)

</div>

---

## 🤔 Wait, what?

LTX-2.3 is a **22-billion-parameter** text-to-video model that generates video **and** a
synchronized audio track at the same time. The official guidance points you at **24–48 GB
datacenter GPUs**. In bf16 the transformer alone is ~38 GB and the text encoder another ~48 GB.

tinycine runs it on a **humble RTX 3060 with 12 GB of VRAM** — the kind of card that's been in
gaming PCs since 2021 — and the VRAM high-water mark comes in at **~2.5 GB**. That's right:
**five times under the ceiling.** 🤯

You type a prompt. About three minutes later you have a real `.mp4` with sound.

```
"a red fox trotting through a snowy forest at dawn, cinematic, soft golden light"
                                   │
                                   ▼
                    🦊  768×768 · 49 frames · with audio
```

## 🚀 Quick start

```bash
# 0) Clone, set up the CUDA venv + deps (one time) — see SETUP.md for the full walkthrough
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt

# 1) Download the model weights (~95 GB, resumable) — go make a coffee ☕
./download_models.sh

# 2) Make a movie 🎬
./runme.sh "a neon-lit alley in the rain, a cat watching from a windowsill, cinematic"
```

Your clip lands in `assets/outputs/tinycine.mp4`. **That's it.**

Want to tweak prompt, quality, seed, or upscaling? Open `runme.sh` (it's heavily commented) or
run `python3 scripts/ltxv_cli.py --help` for every knob.

## 🪄 How on earth does this fit?

No magic — just every memory trick in the book, stacked, and **measured** (not guessed):

| Trick | What it does |
|-------|--------------|
| 🧩 **Two-phase loading** | The Gemma-3-12B text encoder and the DiT transformer are the two giants — and they **never sit in memory at the same time**. Phase A loads Gemma (in 4-bit), encodes your prompt, then frees it. Phase B loads the transformer and denoises. |
| 🌊 **Sequential CPU offload** | Only the layer being computed lives on the GPU; the rest streams from your 62 GB of system RAM, block by block. |
| 🗜️ **fp8 / 4-bit weights** | The transformer is stored in fp8 and the text encoder in 4-bit, then upcast to bf16 for the actual math. (Ampere has no fp8 tensor cores — so the win here is *memory*, not speed.) |
| 🧱 **Tiled VAE decode** | Decoding latents to pixels is the single biggest VRAM spike. Tiling + temporal chunking flattens it, so the peak barely moves even at 1024². |
| ⚡ **Distilled 8-step schedule** | LTX-2.3-Distilled gets there in 8 denoising steps instead of 30+. |

The result: a peak that's **dominated by the streamed weights, not the resolution** — so going
from 512² to 1024² costs you *time*, not *VRAM*.

## 🎚️ Pick your preset

`runme.sh` defaults to **balanced** — the sweet spot tuned for the 3060. All three presets were
measured on the real card (RTX 3060 12 GB, driver 580.x, CUDA 12.9, torch 2.6.0+cu124):

| Preset | Resolution × frames | VRAM peak | Wall time | Best for |
|--------|--------------------:|----------:|----------:|----------|
| `fast` | 512×512 × 25 | **1.94 GB** | ~105 s | drafts, fast iteration |
| **`balanced`** (default) | 768×768 × 49 | **2.52 GB** | ~3.4 min | the everyday clip |
| `high` | 1024×1024 × 49 | **2.55 GB** | ~4.8 min | hero shots |

> Every preset stays **~5× under the 12 GB ceiling**, and host RAM returns to baseline after each
> run. Switch presets by editing `QUALITY=` near the top of `runme.sh`.

### 🔍 Optional: two-stage upscale

Generate small and fast, then sharpen with LTX-2.3's own latent upsamplers — at the *same* VRAM:

```bash
python3 scripts/ltxv_cli.py \
    --prompt "a red fox in fresh snow, golden hour backlight" \
    --size 512x512 --frames 49 --upscale spatial --keep-base \
    --out assets/outputs/fox_4k.mp4
```

`spatial` doubles resolution (512²→1024²), `temporal` doubles the frame count, and
`spatial+temporal` chains both.

## 📦 What's in the box

```
tinycine/
├── runme.sh              ▶ one-command generator (start here)
├── download_models.sh    ⬇ fetch the LTX-2.3 weights into models/ (resumable)
├── requirements.txt      📋 Python deps (install torch first — see SETUP.md)
├── SETUP.md              🛠 full install + troubleshooting walkthrough
├── scripts/
│   ├── ltxv_cli.py             🎬 the generator (two-phase load, all the knobs)
│   ├── fetch_diffusers_repo.py ⬇ resumable model fetch with progress
│   ├── contact_sheet.py        🖼 QA helper: tile/extract frames from clips
│   └── inspect_safetensors.py  🔬 torch-free safetensors header dump
└── assets/outputs/       🎞 your generated clips land here
```

Model weights are **not** included (they're ~95 GB and © Lightricks) — `download_models.sh`
fetches them. Nothing here redistributes any weights.

## 🧰 Requirements (TL;DR)

- **GPU:** NVIDIA Ampere (sm_86+), 12 GB — the RTX 3060 12 GB is the reference card.
- **System RAM:** ~62 GB (it holds the offloaded weights).
- **Disk:** ~100 GB free for the model repo.
- **Software:** Linux, recent NVIDIA driver, CUDA 12.x, Python 3.10+.

Full details, exact install commands, and troubleshooting → **[SETUP.md](SETUP.md)**.

## ❤️ License & credits

- **tinycine source code:** MIT © **g023** — see [LICENSE](LICENSE). Hack away.
- **LTX-2.3 model weights:** © **[Lightricks](https://www.lightricks.com/)** under the LTX-2
  Community License. tinycine **does not** contain or redistribute them; `download_models.sh`
  pulls them from the official Hugging Face repos. All credit for the model goes to Lightricks.
- Built on the wonderful [🤗 diffusers](https://github.com/huggingface/diffusers) library.

---

<div align="center">

*Made for everyone whose GPU was "too small" to play. Turns out it wasn't.* ✨

**If tinycine made you something cool, star the repo and tag [@g023](https://github.com/g023).**

</div>
