"""No-op interpreter shim for the vendored LoRC trainer.

PAAS_LoRC's version re-execs into the shared PAAS_qwen3vl venv, because that is the only Python on
that host with a new enough transformers. Inside PAAS_v4_full that behaviour would be WRONG and
silent: run_finetuning.sh deliberately pins this project's own venv (see the SELF-CONTAINED VENV
note at the top of it), and re-execing elsewhere would run training against a different stack than
the one the rest of the pipeline uses.

So the hand-off is a no-op here, and the interpreter stays whatever the caller chose. The import
is kept so the vendored train.py needs no edit -- a deleted symbol would be a merge conflict every
time the trainer is refreshed from upstream.
"""


def ensure_interpreter(*_a, **_k):
    return None
