"""Single source of truth for every path used by the V-cache layout modify chain.

Source models live in tinyllama2_110M_qmode2_new/ and are never written to.
Outputs go to this folder's parent (tinyllama2_110M_qmode2_new_vmove/), mirroring
the source layout so the yaml sidecars sit next to the model AnDLA compiles.

Override either root with an env var when experimenting:
  VMOVE_SRC_ROOT / VMOVE_OUT_ROOT
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_OUT_ROOT = os.path.dirname(_HERE)

# SRC_ROOT is anchored to this file's location, NOT to OUT_ROOT: overriding the
# output root must not move the source root with it.
SRC_ROOT = os.environ.get(
    "VMOVE_SRC_ROOT",
    os.path.join(os.path.dirname(_DEFAULT_OUT_ROOT), "tinyllama2_110M_qmode2_new"),
)
OUT_ROOT = os.environ.get("VMOVE_OUT_ROOT", _DEFAULT_OUT_ROOT)

DECODE_SRC = os.path.join(SRC_ROOT, "decode", "model.tflite")
DECODE_OUT = os.path.join(OUT_ROOT, "decode", "model.tflite")
PREFILL_SRC = os.path.join(SRC_ROOT, "prefill_128", "model.tflite")
PREFILL_OUT = os.path.join(OUT_ROOT, "prefill_128", "model.tflite")
