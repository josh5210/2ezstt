from faster_whisper import WhisperModel

import os

EZSTT_MODEL_DIR = os.getenv("EZSTT_MODEL_DIR")

try:
    m = WhisperModel("medium", device="cuda", compute_type="float16", download_root=EZSTT_MODEL_DIR)
    print("Loaded on CUDA ✔")
except Exception as e:
    print("CUDA failed, falling back to CPU:", e)
    m = WhisperModel("medium", device="cpu", compute_type="int8", download_root=EZSTT_MODEL_DIR)
print("OK:", m is not None)
