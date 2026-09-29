"""Path resolution for the vendored LoRC trainer, rooted in PAAS_v4_full.

DELIBERATELY NOT A COPY of PAAS_LoRC/lorc/paths.py. That file resolves `data/`, `base_models/`
and `runs/` against the PAAS_LoRC tree, and it special-cases an external 26 GB encoder. Copying it
here would have this project's training read manifests and write checkpoints into a DIFFERENT
project -- which would work on this machine, produce plausible output, and be wrong.

Only the two functions the vendored modules actually call are provided: resolve() and remap().
Anything else they might reach for should fail loudly rather than resolve to a surprising place.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # <v4_full>/lorc
PROJECT_ROOT = os.path.dirname(ROOT)                                        # <v4_full>
DATA = os.environ.get("LORC_DATA", os.path.join(PROJECT_ROOT, "data"))
BASE_MODELS = os.environ.get("LORC_BASE_MODELS", os.path.join(PROJECT_ROOT, "base_models"))
RUNS = os.environ.get("LORC_RUNS", os.path.join(PROJECT_ROOT, "runs", "lorc"))

# Only set when the image corpus has moved relative to what the manifests record.
MANIFEST_IMAGE_ROOT = os.environ.get("LORC_MANIFEST_IMAGE_ROOT", "/datasets/work/vLLM/data")
IMAGE_ROOT = os.environ.get("LORC_IMAGE_ROOT", "")


def remap(image_path):
    """Rewrite a manifest image path if LORC_IMAGE_ROOT says the corpus moved."""
    if not IMAGE_ROOT or not image_path.startswith(MANIFEST_IMAGE_ROOT):
        return image_path
    return IMAGE_ROOT.rstrip("/") + image_path[len(MANIFEST_IMAGE_ROOT):]


def resolve(p):
    """Absolute path for a config value: absolute stays, relative resolves against this project.

    A relative path resolves against its OWN override (LORC_DATA / LORC_BASE_MODELS / LORC_RUNS)
    rather than blindly against the root, so moving one of them cannot leave the pre-flight and
    the trainer reading different files while both look correct.
    """
    if not p:
        return p
    p = os.path.expanduser(str(p))
    if os.path.isabs(p):
        return p
    norm = p.replace(os.sep, "/")
    for prefix, base in (("data/", DATA), ("base_models/", BASE_MODELS), ("runs/", RUNS)):
        if norm == prefix.rstrip("/"):
            return base
        if norm.startswith(prefix):
            return os.path.normpath(os.path.join(base, norm[len(prefix):]))
    return os.path.normpath(os.path.join(PROJECT_ROOT, norm))
