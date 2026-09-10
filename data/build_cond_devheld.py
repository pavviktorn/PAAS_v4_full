#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Filter the DEV holdout out of the MLLM's CONDITIONING-DISTILLATION corpus.

WHY THIS FILE EXISTS SEPARATELY FROM build_effaa_devheld.py. The MLLM is trained in two passes and
they read DIFFERENT corpora:

  qwen_lora    --train $EFFAA_TRAIN  -> eFFAA_ext.json          (filtered by build_effaa_devheld.py)
  qwen_distill --train $QWEN_COND    -> effaa_cond_mix.json      <- NOT covered by that filter

effaa_cond_mix.json is a pre-built file (2026-07-18, 700,000 records) whose images span several
roots. 94,108 of its distinct images sit directly under the MIDS root, so unlike the eFFAA case the
leak is visible by EXACT PATH:

  es_dev_sel        5,669/53,444 = 10.61%   (the selector)
  es_dev_eval_c99   4,903/35,171 = 13.94%   (the final report split)

That is larger than the eFFAA leak it sits next to (7.73% / 5.13%), and it is REAL-SKEWED: 60% of
the affected certified rows are real against 35% real in the split, and real is the class that sets
the operating threshold. Every non-MLLM member has 0% of DEV, so the inflation does not cancel in a
comparison -- it advantages exactly one member (`ffaa`).

CONTENT HASHING TOO, not just paths. 431k of its images live under other roots
(/datasets/newout/PAD/..., /datasets/work/vLLM/data/fmt_error_all, ...). The eFFAA audit showed the
same pictures get re-filed under different names across roots, invisible to path comparison, so
paths alone are not sufficient evidence of cleanliness here.

WRITES UNDER temp/ ON PURPOSE. The source is shared and outside the working tree.
"""
import hashlib
import json
import os
from multiprocessing import Pool

SRC = "/datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen/effaa_cond_mix.json"
OUT = "/datasets/work/vLLM/temp/PAAS_v4_full/data/effaa_cond_mix_devheld.json"
ES = "/datasets/work/vLLM/temp/EVAL_SPACE"
SCRATCH = "/tmp/claude-1001/-datasets-work-vLLM-temp/78771557-5bd2-4eaf-81fb-4801ebb47561/scratchpad"
CACHE = f"{SCRATCH}/cond_hashes.tsv"
NPROC = 24


def md5(p):
    h = hashlib.md5()
    try:
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None


def main():
    dev_h, dev_p = set(), set()
    for line in open(f"{SCRATCH}/dev_hashes.tsv"):
        h, p = line.rstrip("\n").split("\t", 1)
        dev_p.add(p)
        if h != "None":
            dev_h.add(h)
    print(f"[cond-filter] DEV rows: {len(dev_p):,} paths, {len(dev_h):,} content hashes", flush=True)

    recs = json.load(open(SRC))
    paths = sorted({r["image"] for r in recs if isinstance(r.get("image"), str)})
    print(f"[cond-filter] {len(recs):,} records, {len(paths):,} distinct images", flush=True)

    cached = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            h, p = line.rstrip("\n").split("\t", 1)
            cached[p] = None if h == "None" else h
        print(f"[cond-filter] reusing {len(cached):,} cached hashes", flush=True)
    need = [p for p in paths if p not in cached]
    if need:
        print(f"[cond-filter] hashing {len(need):,} images with {NPROC} procs ...", flush=True)
        done = 0
        with Pool(NPROC) as pool:
            for p, h in zip(need, pool.imap(md5, need, chunksize=256)):
                cached[p] = h
                done += 1
                if done % 100_000 == 0:
                    print(f"[cond-filter]   {done:,}/{len(need):,}", flush=True)
        with open(CACHE, "w") as fh:
            for p, h in cached.items():
                fh.write(f"{h}\t{p}\n")

    keep, drop_path, drop_hash = [], 0, 0
    for r in recs:
        p = r.get("image")
        if isinstance(p, str):
            if p in dev_p:
                drop_path += 1
                continue
            if cached.get(p) in dev_h:
                drop_hash += 1
                continue
        keep.append(r)

    with open(OUT, "w") as fh:
        json.dump(keep, fh)
    # verify: nothing kept resolves to DEV by either route
    bad = sum(1 for r in keep
              if isinstance(r.get("image"), str)
              and (r["image"] in dev_p or cached.get(r["image"]) in dev_h))
    if bad:
        raise SystemExit(f"[cond-filter] REFUSING: {bad:,} DEV rows survived")

    rep = {"what": "conditioning-distillation corpus with the DEV holdout removed",
           "src": SRC, "out": OUT, "n_in": len(recs), "n_out": len(keep),
           "dropped_by_path": drop_path, "dropped_by_content": drop_hash,
           "pct_dropped": round((drop_path + drop_hash) / len(recs) * 100, 3),
           "verified_dev_rows_remaining": 0}
    json.dump(rep, open("/datasets/work/vLLM/temp/PAAS_v4_full/data/cond_devheld_provenance.json", "w"), indent=1)
    print(f"[cond-filter] {len(recs):,} -> {len(keep):,}  "
          f"(dropped {drop_path:,} by path + {drop_hash:,} by content = "
          f"{(drop_path+drop_hash)/len(recs)*100:.3f}%)", flush=True)
    print(f"[cond-filter] verified 0 DEV rows remain -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
