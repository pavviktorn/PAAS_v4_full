#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Filter the DEV split out of the MLLM's eFFAA corpus, by CONTENT.

WHY --exclude CANNOT DO THIS. --exclude operates on the MIDS image root
(/datasets/work/vLLM/data/no_delete_mids_train). The MLLM trains on a different corpus rooted at
/datasets/newout, with different directory structure AND different filenames for the same pictures:

  /datasets/newout/3D_ATTACKS/mis-classified_4+13fmt_allpad/fake/pad/0002_3.jpg
    is byte-identical to
  .../no_delete_mids_train/3D_ATTACKS/pad/HiFi_Mask/fake/train/1_83_3_6_6_2/0002.jpg

Path or basename comparison finds nothing (measured: 0 path overlap). Content hashing finds 8,963
corpus images that ARE DEV images -- 7,743 distinct DEV rows, i.e. 8.40% of the holdout.

WHY IT MATTERS ENOUGH TO FIX. Every other member has 0% of DEV. So this is not an inflation that
cancels when members are compared; it advantages exactly one member (`ffaa`) in the cross-model
ranking the combination search performs. It reaches 7.73% of es_dev_sel (the selector) and 5.13% of
es_dev_eval_c99 (the final report -- c99 cut near-duplicates against the MIDS trainset and knows
nothing about this corpus). It is also REAL-SKEWED: 46% of the affected certified rows are real
against 35% real in the split, and real is the class that sets the operating threshold.

COST OF THE FIX: 0.67% of the corpus. Negligible for MLLM training.

WRITES UNDER temp/ ON PURPOSE. The source lives in a shared location outside the working tree;
rewriting it in place to work around a config problem would be very hard to undo and would silently
change every other consumer of that file.
"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool

IR = "/datasets/newout"
SRC_DIR = f"{IR}/vqa_info_2+13+4+3_fmt"
OUT_DIR = "/datasets/work/vLLM/temp/PAAS_v4_full/data"
SCRATCH = "/tmp/claude-1001/-datasets-work-vLLM-temp/78771557-5bd2-4eaf-81fb-4801ebb47561/scratchpad"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-hashes", default=f"{SCRATCH}/dev_hashes.tsv")
    ap.add_argument("--effaa-hashes", default=f"{SCRATCH}/effaa_hashes.tsv",
                    help="reused if it covers the corpus; otherwise recomputed")
    a = ap.parse_args()

    dev = set()
    for line in open(a.dev_hashes):
        h = line.split("\t", 1)[0]
        if h != "None":
            dev.add(h)
    print(f"[effaa-filter] DEV content hashes: {len(dev):,}", flush=True)

    cached = {}
    if os.path.exists(a.effaa_hashes):
        for line in open(a.effaa_hashes):
            h, p = line.rstrip("\n").split("\t", 1)
            cached[p] = None if h == "None" else h
        print(f"[effaa-filter] reusing {len(cached):,} cached corpus hashes", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"what": "eFFAA corpus with the DEV holdout removed by content",
              "source_dir": SRC_DIR, "out_dir": OUT_DIR, "files": {}}

    for name in ("eFFAA_ext.json", "eFFAA_ext_eval.json"):
        src = os.path.join(SRC_DIR, name)
        recs = json.load(open(src))
        paths = sorted({os.path.join(IR, r["image"]) for r in recs
                        if isinstance(r.get("image"), str)})
        need = [p for p in paths if p not in cached]
        if need:
            print(f"[effaa-filter] {name}: hashing {len(need):,} uncached images", flush=True)
            with Pool(NPROC) as pool:
                for p, h in zip(need, pool.imap(md5, need, chunksize=256)):
                    cached[p] = h

        keep, drop = [], 0
        for r in recs:
            p = r.get("image")
            full = os.path.join(IR, p) if isinstance(p, str) else None
            if full is not None and cached.get(full) in dev:
                drop += 1
                continue
            keep.append(r)

        out = os.path.join(OUT_DIR, name.replace(".json", "_devheld.json"))
        with open(out, "w") as fh:
            json.dump(keep, fh)
        report["files"][name] = {"src": src, "out": out, "n_in": len(recs),
                                 "n_dropped": drop, "n_out": len(keep),
                                 "pct_dropped": round(drop / max(len(recs), 1) * 100, 3)}
        print(f"[effaa-filter] {name}: {len(recs):,} -> {len(keep):,} "
              f"(dropped {drop:,} = {drop/max(len(recs),1)*100:.3f}%) -> {out}", flush=True)

    # verify: no kept record resolves to DEV content
    for name, info in report["files"].items():
        recs = json.load(open(info["out"]))
        bad = sum(1 for r in recs
                  if cached.get(os.path.join(IR, r["image"])) in dev)
        info["verified_dev_rows_remaining"] = bad
        if bad:
            raise SystemExit(f"[effaa-filter] REFUSING: {bad:,} DEV rows survived in {name}")
        print(f"[effaa-filter] verified {name}: 0 DEV rows remain", flush=True)

    with open(f"{OUT_DIR}/effaa_devheld_provenance.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[effaa-filter] wrote {OUT_DIR}/effaa_devheld_provenance.json", flush=True)


if __name__ == "__main__":
    main()
