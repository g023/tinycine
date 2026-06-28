#!/usr/bin/env python3
"""ltxv_cli.py - LTX-2.3 text->video on a 12 GB RTX 3060 via diffusers LTX2Pipeline.

Strategy (see kb/engine_decision.md, research/02_memory_strategy.md):
  * Two-phase, host-RAM-safe: (A) load Gemma text encoder in 4-bit, encode the prompt(s)
    to prompt_embeds, FREE it; (B) load the LTX2 transformer + VAE (text_encoder=None),
    denoise + decode. Gemma (bf16 = 48 GB) and the transformer (bf16 = 38 GB) are never
    co-resident in the 62 GB host RAM -> avoids the WHY_I_CRASHED pinned-RAM OOM.
  * VRAM levers: sequential CPU offload, fp8 layerwise casting on the transformer
    (Ampere has no fp8 MMA -> storage only, upcast to bf16 for compute), VAE tiling,
    distilled 8-step / guidance 1.0.
  * No silent truncation: every memory lever / reduction is logged.

All heavy phases print periodic progress.
"""
import argparse
import gc
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR_DEFAULT = os.path.join(REPO_ROOT, "models", "diffusers-distilled")
UPSCALER_DIR_DEFAULT = os.path.join(REPO_ROOT, "models", "upscalers")

# Single-file upscaler checkpoints (Kijai/ComfyUI mirror) -> LTX2LatentUpsamplerModel.
# Configs inferred from the safetensors key shapes (verified by strict load_state_dict):
#   spatial : initial_conv [1024,128,3,3,3] (mid=1024, dims=3), upsampler.0 [4096,1024,3,3]
#             -> Conv2d(mid,4*mid)+PixelShuffleND(2), i.e. non-rational spatial x2.
#   temporal: initial_conv [512,128,3,3,3] (mid=512, dims=3), upsampler.0 [1024,512,3,3,3]
#             -> Conv3d(mid,2*mid)+PixelShuffleND(1), i.e. temporal x2.
UPSCALERS = {
    "spatial": (
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        dict(in_channels=128, mid_channels=1024, num_blocks_per_stage=4, dims=3,
             spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False),
    ),
    "temporal": (
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        dict(in_channels=128, mid_channels=512, num_blocks_per_stage=4, dims=3,
             spatial_upsample=False, temporal_upsample=True),
    ),
}


# ----------------------------------------------------------------------------- utils
def log(msg):
    print(f"[ltxv {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    return None


def host_rss_gb():
    # resident set of this process tree (approx host-RAM high-water proxy)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


def vram_peak_gb():
    import torch
    return torch.cuda.max_memory_allocated() / 1e9


def vram_reserved_gb():
    import torch
    return torch.cuda.max_memory_reserved() / 1e9


# ----------------------------------------------------------------------------- quality encode
def encode_video_q(video, fps, audio, audio_sample_rate, output_path,
                   crf=18, preset="slow", pix_fmt="yuv420p"):
    """Quality-controlled re-implementation of diffusers' ltx2 encode_video.

    The stock diffusers encode_video (export_utils.py) adds a libx264 stream but sets NO
    crf/bitrate, so PyAV falls back to libx264's ~1 Mbps default -> visible blocking/banding
    on top of the model output (~200 KB for a 2 s 512^2 clip). We keep our copy here (do NOT
    edit the installed package) and expose crf/preset/pix_fmt. Audio path reuses the diffusers
    helpers unchanged. crf 18 is visually near-lossless; lower = bigger/better, ~18-23 sane.
    """
    import av
    import numpy as np
    import torch
    import PIL.Image
    from diffusers.pipelines.ltx2.export_utils import _prepare_audio_stream, _write_audio

    # ---- normalize input to a uint8 tensor [F, H, W, C] (mirror diffusers' coercion)
    if isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
        video = torch.from_numpy(np.stack([np.array(f) for f in video], axis=0))
    elif isinstance(video, np.ndarray):
        is_denorm = np.logical_and(video >= 0.0, video <= 1.0)
        if np.all(is_denorm):
            video = (video * 255).round().astype("uint8")
        video = torch.from_numpy(video)
    elif isinstance(video, list):  # list of np frames (e.g. from upscaler)
        arr = np.stack([np.asarray(f) for f in video], axis=0)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype("uint8")
        video = torch.from_numpy(arr)

    _, height, width, _ = video.shape
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    # the whole point: tell libx264 to keep detail instead of its ~1 Mbps CRF~23+ default
    stream.options = {"crf": str(crf), "preset": preset}

    audio_stream = None
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")
        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    for frame_array in video.to("cpu").numpy():
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)
    container.close()
    log(f"encoded {output_path} (libx264 crf={crf} preset={preset} {pix_fmt} "
        f"{width}x{height} {fps}fps)")


# ----------------------------------------------------------------------------- phase A
def encode_prompts(repo, prompts, neg_prompt, device, max_seq_len, text_encoder_4bit=True):
    """Load Gemma (4-bit by default), produce stacked-hidden-state prompt_embeds, free it.

    Replicates LTX2Pipeline._get_gemma_prompt_embeds: tokenize (left pad), forward with
    output_hidden_states, stack all layers on the last dim, flatten(2,3) to 3D.
    """
    import torch
    from transformers import AutoTokenizer

    te_path = os.path.join(repo, "text_encoder")
    tok_path = os.path.join(repo, "tokenizer")
    log(f"Phase A: loading Gemma text encoder (4bit={text_encoder_4bit}) from {te_path}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import Gemma3ForConditionalGeneration
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if text_encoder_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = {"": 0}
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(te_path, **load_kwargs)
    if not text_encoder_4bit:
        text_encoder = text_encoder.to(device)
    text_encoder.eval()
    log(f"Phase A: Gemma loaded in {time.time()-t0:.1f}s; host RSS {host_rss_gb():.1f} GB, "
        f"VRAM {vram_peak_gb():.2f} GB")

    def embed(texts):
        texts = [t.strip() for t in texts]
        ti = tokenizer(texts, padding="max_length", max_length=max_seq_len,
                       truncation=True, add_special_tokens=True, return_tensors="pt")
        ids = ti.input_ids.to(device)
        mask = ti.attention_mask.to(device)
        with torch.no_grad():
            out = text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=-1)          # [B, T, D, L+1]
        emb = hs.flatten(2, 3).to(dtype=torch.bfloat16)      # [B, T, D*(L+1)]
        return emb.cpu(), mask.cpu()

    pe, pm = embed(prompts)
    log(f"Phase A: prompt_embeds {tuple(pe.shape)} dtype {pe.dtype}")
    npe = npm = None
    if neg_prompt is not None:
        npe, npm = embed([neg_prompt] * len(prompts))
        log(f"Phase A: negative_prompt_embeds {tuple(npe.shape)}")

    # free Gemma fully before loading the transformer
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log(f"Phase A done in {time.time()-t0:.1f}s; Gemma freed. "
        f"host MemAvailable now {mem_available_gb():.1f} GB")
    return pe, pm, npe, npm


# ----------------------------------------------------------------------------- phase B
def build_denoise_pipeline(repo, device, fp8_transformer=True, offload="sequential",
                           vae_tiling=True):
    """Load LTX2Pipeline WITHOUT the text encoder; engage VRAM levers."""
    import torch
    from diffusers import LTX2Pipeline

    log(f"Phase B: loading LTX2Pipeline (transformer+vae, text_encoder=None) from {repo}")
    t0 = time.time()
    pipe = LTX2Pipeline.from_pretrained(
        repo, text_encoder=None, tokenizer=None, processor=None,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    log(f"Phase B: pipeline loaded in {time.time()-t0:.1f}s; host RSS {host_rss_gb():.1f} GB")

    levers = []
    if fp8_transformer:
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        levers.append("transformer fp8 layerwise-casting (storage e4m3 / compute bf16)")
    if vae_tiling:
        pipe.vae.enable_tiling()
        levers.append("vae.enable_tiling()")
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        levers.append("enable_sequential_cpu_offload()")
    elif offload == "model":
        pipe.enable_model_cpu_offload()
        levers.append("enable_model_cpu_offload()")
    else:
        pipe.to(device)
        levers.append(f"no offload (pipe.to({device}))")
    for lv in levers:
        log(f"Phase B: lever engaged -> {lv}")
    return pipe


# ----------------------------------------------------------------------------- phase D (upscale)
def load_upsampler(kind, upscaler_dir, dtype):
    """Instantiate LTX2LatentUpsamplerModel and strict-load the single-file checkpoint."""
    import torch
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel
    from safetensors.torch import load_file

    fname, cfg = UPSCALERS[kind]
    path = os.path.join(upscaler_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{kind} upscaler not found at {path}")
    model = LTX2LatentUpsamplerModel(**cfg)
    sd = load_file(path)
    model.load_state_dict(sd, strict=True)          # strict -> validates the inferred config
    model = model.to(dtype=dtype).eval()
    log(f"Phase D: loaded {kind} upsampler ({len(sd)} tensors) from {os.path.basename(path)}")
    return model


def run_upscale(video, kind, repo, upscaler_dir, in_w, in_h, in_frames, fps, seed,
                device, offload="sequential", vae_tiling=True):
    """Spatially/temporally upscale stage-1 frames via LTX2LatentUpsamplePipeline.

    `video` is a list of PIL frames (the low-res stage-1 output). The pipeline re-encodes
    them with the video VAE, runs the latent upsampler, and decodes back to RGB. A fresh VAE
    is loaded (the stage-1 pipeline is freed first) so accelerate offload hooks don't collide.
    Returns (frames_np[F,H,W,C] in [0,1], out_w, out_h).
    """
    import torch
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
    from diffusers.models import AutoencoderKLLTX2Video

    t0 = time.time()
    log(f"Phase D: {kind} upscale of {in_w}x{in_h}x{in_frames} starting; "
        f"loading VAE + upsampler")
    vae = AutoencoderKLLTX2Video.from_pretrained(
        repo, subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    upsampler = load_upsampler(kind, upscaler_dir, torch.bfloat16)
    up = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=upsampler)

    levers = []
    if vae_tiling:
        up.vae.enable_tiling(); levers.append("vae.enable_tiling()")
    if offload == "sequential":
        up.enable_sequential_cpu_offload(); levers.append("enable_sequential_cpu_offload()")
    elif offload == "model":
        up.enable_model_cpu_offload(); levers.append("enable_model_cpu_offload()")
    else:
        up.to(device); levers.append(f"no offload (to({device}))")
    for lv in levers:
        log(f"Phase D: lever engaged -> {lv}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = up(video=video, height=in_h, width=in_w, num_frames=in_frames,
             generator=gen, output_type="np", return_dict=False)[0]
    frames = out[0]                                  # [F, H, W, C] in [0,1]
    out_h, out_w = frames.shape[1], frames.shape[2]
    log(f"Phase D: {kind} upscale done in {time.time()-t0:.1f}s -> {out_w}x{out_h}x"
        f"{frames.shape[0]}  VRAM peak {vram_peak_gb():.2f} GB")

    del up, vae, upsampler
    gc.collect(); torch.cuda.empty_cache()
    return frames, out_w, out_h


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LTX-2.3 t2v on a 12 GB RTX 3060")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=None, help="default: diffusers DEFAULT_NEGATIVE_PROMPT")
    ap.add_argument("--no-distilled-sigmas", action="store_true",
                    help="do not pass the distilled fixed sigma schedule")
    ap.add_argument("--quality", choices=["fast", "balanced", "high", "none"], default="balanced",
                    help="preset bundling size/frames/cfg/upscale (Q6); explicit flags override. "
                         "'none' = legacy raw defaults (512x512, 49f, cfg1.0)")
    ap.add_argument("--size", default=None, help="WxH, both divisible by 32 (overrides --quality)")
    ap.add_argument("--frames", type=int, default=None, help="must be 8k+1 (overrides --quality)")
    ap.add_argument("--steps", type=int, default=8, help="distilled default 8")
    ap.add_argument("--cfg", type=float, default=None, help="guidance scale (overrides --quality); distilled best 1.0-2.0")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--crf", type=int, default=18,
                    help="libx264 CRF (lower=better/bigger; 18 near-lossless, 14-23 sane)")
    ap.add_argument("--x264-preset", default="slow", help="libx264 speed/efficiency preset")
    ap.add_argument("--pix-fmt", default="yuv420p", help="pixel format (yuv420p compat / yuv444p)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO_DIR_DEFAULT)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--offload", choices=["sequential", "model", "none"], default="sequential")
    ap.add_argument("--no-fp8", action="store_true", help="disable transformer fp8 layerwise casting")
    ap.add_argument("--no-vae-tiling", action="store_true")
    ap.add_argument("--te-bf16", action="store_true", help="load Gemma in bf16 instead of 4-bit (needs RAM)")
    ap.add_argument("--min-ram-gb", type=float, default=8.0)
    ap.add_argument("--upscale", default=None,
                    choices=["none", "spatial", "temporal", "spatial+temporal"],
                    help="stage-2 latent upscale: spatial x2 and/or temporal x2 (overrides --quality)")
    ap.add_argument("--upscaler-dir", default=UPSCALER_DIR_DEFAULT)
    ap.add_argument("--keep-base", action="store_true",
                    help="when upscaling, also keep the low-res stage-1 mp4 (<out>.base.mp4)")
    args = ap.parse_args()

    # ----- resolve --quality preset (Q6 winners); explicit flags above still override.
    # Evidence (logs/tuning_notes.md): native resolution buys real detail at flat 1.94 GB VRAM;
    # cfg 1.5-2.0 adds contrast/adherence (cfg>1 ~+40% wall); fp8 == bf16 quality but faster;
    # 8-step distilled is the right schedule. fast trades frames for <90 s drafts.
    PRESETS = {
        "fast":     dict(size="512x512",   frames=25, cfg=1.0, upscale="none"),
        "balanced": dict(size="768x768",   frames=49, cfg=1.5, upscale="none"),
        "high":     dict(size="1024x1024", frames=49, cfg=2.0, upscale="none"),
        "none":     dict(size="512x512",   frames=49, cfg=1.0, upscale="none"),  # legacy raw
    }
    preset = PRESETS[args.quality]
    applied = []
    for k in ("size", "frames", "cfg", "upscale"):
        if getattr(args, k) is None:
            setattr(args, k, preset[k])
            applied.append(f"{k}={preset[k]}")
        else:
            applied.append(f"{k}={getattr(args, k)}*")  # * = explicit override
    log(f"--quality {args.quality} -> {', '.join(applied)}  (* = CLI override)")

    # ----- validate dims (loud, no silent fixups)
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except Exception:
        ap.error(f"--size must be WxH, got {args.size!r}")
    if w % 32 or h % 32:
        ap.error(f"width/height must be divisible by 32; got {w}x{h}")
    if (args.frames - 1) % 8:
        ap.error(f"--frames must be 8k+1 (e.g. 25, 49, 121); got {args.frames}")

    import torch
    if not torch.cuda.is_available():
        log("FATAL: CUDA not available"); sys.exit(2)
    device = "cuda"

    # ----- host RAM preflight (WHY_I_CRASHED safety)
    avail = mem_available_gb()
    log(f"Preflight: MemAvailable {avail:.1f} GB, VRAM total "
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB on "
        f"{torch.cuda.get_device_name(0)}")
    if avail < args.min_ram_gb:
        log(f"FATAL: only {avail:.1f} GB host RAM available (< {args.min_ram_gb}); "
            f"refusing to load. If the box is in the leaked state, sudo reboot (WHY_I_CRASHED.md).")
        sys.exit(3)
    if not os.path.isdir(args.repo):
        log(f"FATAL: weights repo not found at {args.repo}"); sys.exit(4)

    log(f"Config: {w}x{h} {args.frames}f {args.steps}steps cfg={args.cfg} seed={args.seed} "
        f"offload={args.offload} fp8={not args.no_fp8} vae_tiling={not args.no_vae_tiling}")

    torch.cuda.reset_peak_memory_stats()
    wall0 = time.time()

    # ----- distilled sigma schedule + default negative
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, DEFAULT_NEGATIVE_PROMPT
    sigmas = None
    if not args.no_distilled_sigmas:
        sigmas = list(DISTILLED_SIGMA_VALUES)
        if args.steps != len(sigmas):
            log(f"NOTE: distilled sigma schedule has {len(sigmas)} steps; overriding "
                f"--steps {args.steps} -> {len(sigmas)} (use --no-distilled-sigmas to free-run)")
            args.steps = len(sigmas)
    neg = args.negative if args.negative is not None else DEFAULT_NEGATIVE_PROMPT
    cfg_on = args.cfg > 1.0
    if not cfg_on:
        log(f"NOTE: guidance_scale={args.cfg} <= 1.0 -> CFG disabled, negative prompt unused; "
            f"skipping negative encode")

    # ===== Phase A: encode prompt, free Gemma
    pe, pm, npe, npm = encode_prompts(
        args.repo, [args.prompt], (neg if cfg_on else None), device, args.max_seq_len,
        text_encoder_4bit=not args.te_bf16)

    # ===== Phase B: load denoise pipeline
    pipe = build_denoise_pipeline(
        args.repo, device, fp8_transformer=not args.no_fp8,
        offload=args.offload, vae_tiling=not args.no_vae_tiling)

    # ===== Phase C: generate
    log("Phase C: denoise + decode starting")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    tgen = time.time()
    step_t = {"n": 0, "last": time.time()}

    def cb(pipe_, step, timestep, kw):
        step_t["n"] += 1
        now = time.time()
        log(f"  step {step_t['n']}/{args.steps}  dt {now-step_t['last']:.1f}s  "
            f"VRAM {vram_peak_gb():.2f} GB")
        step_t["last"] = now
        return kw

    upscaling = args.upscale != "none"
    out_type = "pil" if upscaling else "np"   # upscaler wants PIL frames; encode_video takes both
    video, audio = pipe(
        prompt_embeds=pe.to(device), prompt_attention_mask=pm.to(device),
        negative_prompt_embeds=(npe.to(device) if npe is not None else None),
        negative_prompt_attention_mask=(npm.to(device) if npm is not None else None),
        width=w, height=h, num_frames=args.frames, frame_rate=args.fps,
        num_inference_steps=args.steps, sigmas=sigmas, guidance_scale=args.cfg,
        generator=gen, output_type=out_type, return_dict=False,
        callback_on_step_end=cb,
    )
    log(f"Phase C: generation done in {time.time()-tgen:.1f}s")

    # ===== export (+ optional stage-2 upscale)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def encode_video(frames, fps, audio, audio_sample_rate, output_path):
        # quality-controlled local encoder (CRF) instead of diffusers' bitrate-less default
        encode_video_q(frames, fps, audio, audio_sample_rate, output_path,
                       crf=args.crf, preset=args.x264_preset, pix_fmt=args.pix_fmt)

    aud = None
    asr = 24000
    if audio is not None:
        aud = audio[0].float().cpu()
        try:
            asr = pipe.vocoder.config.output_sampling_rate
        except Exception:
            pass

    final_w, final_h, final_frames, final_fps = w, h, args.frames, args.fps
    if upscaling:
        base_frames = video[0]                       # list of PIL frames (low-res)
        if args.keep_base:
            base_path = args.out + ".base.mp4"
            encode_video(base_frames, fps=args.fps, audio=aud, audio_sample_rate=asr,
                         output_path=base_path)
            log(f"WROTE base (low-res) {base_path}")
        # free the stage-1 pipeline before loading the upscaler (host-RAM + hook hygiene)
        del pipe
        gc.collect(); torch.cuda.empty_cache()
        stages = args.upscale.split("+")
        cur_w, cur_h, cur_frames = w, h, args.frames
        frames = base_frames
        for st in stages:
            from PIL import Image as _Image
            if not isinstance(frames[0], _Image.Image):   # np [F,H,W,C] -> PIL for next stage
                frames = [_Image.fromarray((f * 255).round().astype("uint8")) for f in frames]
            frames, cur_w, cur_h = run_upscale(
                frames, st, args.repo, args.upscaler_dir, cur_w, cur_h, cur_frames,
                args.fps, args.seed, device,
                offload=args.offload, vae_tiling=not args.no_vae_tiling)
            cur_frames = frames.shape[0]
        # temporal upscale changes frame count at fixed fps -> longer (slow-mo) clip
        if "temporal" in stages and cur_frames != args.frames:
            log(f"NOTE: temporal upscale changed frame count {args.frames} -> {cur_frames} "
                f"(played at {args.fps} fps the clip is longer)")
        final_w, final_h, final_frames = cur_w, cur_h, cur_frames
        encode_video(frames, fps=args.fps, audio=aud, audio_sample_rate=asr,
                     output_path=args.out)
    else:
        encode_video(video[0], fps=args.fps, audio=aud, audio_sample_rate=asr,
                     output_path=args.out)

    wall = time.time() - wall0
    peak = vram_peak_gb()
    log(f"WROTE {args.out}  ({final_w}x{final_h}x{final_frames}"
        f"{', upscale='+args.upscale if upscaling else ''})")
    log(f"MEASURED: wall {wall:.1f}s ({wall/final_frames:.2f}s/frame)  "
        f"VRAM peak alloc {peak:.2f} GB / reserved {vram_reserved_gb():.2f} GB  "
        f"host RSS {host_rss_gb():.1f} GB  MemAvailable {mem_available_gb():.1f} GB")
    if peak >= 12.0:
        log(f"WARNING: VRAM peak {peak:.2f} GB >= 12 GB ceiling")
    print(f"RESULT out={args.out} size={final_w}x{final_h}x{final_frames} "
          f"upscale={args.upscale} vram_peak_gb={peak:.2f} wall_s={wall:.1f} "
          f"s_per_frame={wall/final_frames:.2f}", flush=True)


if __name__ == "__main__":
    main()
