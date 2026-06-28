#!/usr/bin/env python3
"""Fetch the diffusers-format LTX-2.3 distilled repo with progress logging.

Idempotent/resumable (huggingface_hub cache). Logs a periodic size delta so a
long download is diagnosable (phase, GB pulled, MB/s, elapsed).
"""
import os, sys, time, threading, subprocess

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "diffusers-distilled")


def dir_size_gb(path):
    try:
        out = subprocess.check_output(["du", "-sb", path], stderr=subprocess.DEVNULL)
        return int(out.split()[0]) / 1e9
    except Exception:
        return 0.0


def monitor(stop):
    t0 = time.time()
    last = dir_size_gb(DEST)
    last_t = t0
    while not stop.is_set():
        time.sleep(20)
        cur = dir_size_gb(DEST)
        now = time.time()
        rate = (cur - last) * 1000 / max(now - last_t, 1e-9)  # MB/s
        print(f"[fetch] {cur:6.1f}/95.0 GB  {rate:6.1f} MB/s  elapsed {int(now-t0):5d}s",
              flush=True)
        last, last_t = cur, now


def main():
    from huggingface_hub import snapshot_download
    os.makedirs(DEST, exist_ok=True)
    print(f"[fetch] repo={REPO}\n[fetch] dest={DEST}\n[fetch] start {time.ctime()}", flush=True)
    stop = threading.Event()
    th = threading.Thread(target=monitor, args=(stop,), daemon=True)
    th.start()
    try:
        path = snapshot_download(repo_id=REPO, local_dir=DEST, repo_type="model",
                                 max_workers=8)
    finally:
        stop.set()
    print(f"[fetch] DONE {time.ctime()}  -> {path}  total {dir_size_gb(DEST):.1f} GB", flush=True)


if __name__ == "__main__":
    main()
