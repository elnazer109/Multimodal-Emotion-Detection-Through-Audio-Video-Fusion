"""One command: pull every finished Kaggle arm, rebuild the results tables, render the PDF.

    python src/make_report.py            # pull from Kaggle, then render
    python src/make_report.py --local    # skip the pull, use results/ as-is

Exists so that finishing takes a minute rather than an evening of hand-editing tables, and so the
numbers in the PDF cannot drift from the CSVs the runs actually wrote. Tables are generated from
the data; the prose around them lives in docs/FINDINGS.md between AUTO markers.
"""
import os, re, subprocess, sys, glob
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINDINGS = os.path.join(REPO, "docs", "FINDINGS.md")
PDF = os.path.join(REPO, "docs", "FINDINGS.pdf")

PAPER = {"audio": 79.24, "video": 92.72, "fusion": 96.06}
PAPER_FOLDS = {"audio": [78.8, 80.6, 77.4, 79.8, 79.6],
               "video": [93.0, 91.9, 93.7, 96.6, 88.2],
               "fusion": [95.6, 97.6, 97.2, 95.3, 94.6]}
NOTEBOOK = {"audio": 80.67, "video": 83.24, "fusion": 94.29}   # its own Table VII, line 897

# kernel -> (results subdir, csv name, human label, what changed)
ARMS = [
    ("ravdess-paper-protocol", "kinetics-arm", "paper_protocol_folds.csv",
     "kinetics", "Kinetics channel stats + scaler"),
    ("ravdess-paper-perclip", "perclip-arm", "paper_perclip_folds.csv",
     "per-clip", "per-clip norm + scaler"),
    ("ravdess-paper-refine", "refine-arm", "paper_refine_folds.csv",
     "+ Fig. 4", "per-clip + scaler + refine head on a real 2x7x7 volume"),
    ("ravdess-paper-epochs", "epochs-arm", "paper_epochs_folds.csv",
     "+ 35 epochs", "Fig. 4 + video 15 -> 35 epochs"),
]
# The honest arm has been re-run as the video model was fixed; prefer the newest results dir.
# actor-independent-kinetics = the first (contaminated) run, kept as an ablation.
HONEST = ("ravdess-actor-independent", "actor-independent-fig4", "fold_results.csv")
HONEST_FALLBACKS = ["actor-independent-fig4", "actor-independent-perclip",
                    "actor-independent-kinetics", "actor-independent"]


def sh(cmd):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)


def save_log(kernel, name):
    """Decode a kernel log to plain text and keep it as evidence.

    Logs are pulled and committed because they are the primary record: per-epoch traces, per-fold
    early-stop points, the sha256 check on the pretrained weights, and the line that prints
    "actors in BOTH train and val: 24/24". They are also PERISHABLE -- re-pushing a kernel makes
    `kernels output` return the new version, so an earlier run's log becomes unrecoverable. One was
    nearly lost that way.
    """
    logs = os.path.join(REPO, "results", "logs")
    os.makedirs(logs, exist_ok=True)
    tmp = os.path.join(logs, "_raw")
    os.makedirs(tmp, exist_ok=True)
    sh(f'python -m kaggle kernels output cubeis/{kernel} -p "{tmp}" --file-pattern ".*\\.log" -o')
    src = glob.glob(os.path.join(tmp, "*.log"))
    if not src:
        return
    import json
    try:
        d = json.load(open(src[0], encoding="utf-8"))
    except Exception:
        return
    lines = [e.get("data", "") for e in (d if isinstance(d, list) else d.get("log", []))]
    keep = [l.rstrip() for l in lines
            if l.strip() and "it/s]" not in l and "?it/s" not in l and "s/it]" not in l]
    open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8").write("\n".join(keep) + "\n")
    for f in src:
        os.remove(f)
    print(f"  {'':28} log -> results/logs/{name}.log ({len(keep)} lines)")


def pull(kernel, subdir):
    """Pull only CSVs and the log -- never the 1.5GB face cache onto a laptop."""
    d = os.path.join(REPO, "results", subdir)
    os.makedirs(d, exist_ok=True)
    r = sh(f'python -m kaggle kernels status cubeis/{kernel}')
    if "COMPLETE" not in r.stdout:
        state = "running" if "RUNNING" in r.stdout else "not available"
        print(f"  {kernel:28} {state} - keeping whatever is already in results/")
        return False
    sh(f'python -m kaggle kernels output cubeis/{kernel} -p "{d}" --file-pattern ".*\\.csv" -o')
    print(f"  {kernel:28} pulled -> results/{subdir}/")
    save_log(kernel, subdir)
    return True


def load(subdir, csv):
    for p in (os.path.join(REPO, "results", subdir, csv),
              *glob.glob(os.path.join(REPO, "results", subdir, "*folds.csv"))):
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def load_honest():
    """Prefer the newest actor-independent run; the older ones stay as ablations."""
    for sub in HONEST_FALLBACKS:
        d = load(sub, "fold_results.csv")
        if d is not None and len(d):
            print(f"  actor-independent: using results/{sub}/")
            return d, sub
    return None, None


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def headline_table(arms):
    rows = ["| arm | what changed | audio | video | fusion | vs paper |",
            "|---|---|---|---|---|---|",
            f"| **paper, Table VII** | *the bar* | {PAPER['audio']} | {PAPER['video']} | "
            f"**{PAPER['fusion']}** | — |",
            f"| notebook, as committed | *never reproduced its own paper* | {NOTEBOOK['audio']} | "
            f"{NOTEBOOK['video']} | {NOTEBOOK['fusion']} | −1.77 |"]
    for label, changed, d in arms:
        p = d.pivot_table(index="fold", columns="model", values="accuracy")
        g = {m: p[m].mean() if m in p else None for m in ["audio", "video", "fusion"]}
        delta = g["fusion"] - PAPER["fusion"] if g["fusion"] is not None else None
        star = " ✅" if delta is not None and delta > 0 else ""
        rows.append(f"| **{label}** | {changed} | {fmt(g['audio'])} | {fmt(g['video'])} | "
                    f"**{fmt(g['fusion'])}** | {fmt(delta):>6}{star} |")
    return "\n".join(rows)


def perfold_table(arms):
    out = []
    for label, _, d in arms:
        p = d.pivot_table(index="fold", columns="model", values="accuracy").round(2)
        hdr = "| fold | " + " | ".join(c for c in ["audio", "video", "fusion"] if c in p) + " |"
        sep = "|---" * (1 + sum(c in p for c in ["audio", "video", "fusion"])) + "|"
        lines = [f"**{label}**", "", hdr, sep]
        for f_, r in p.iterrows():
            lines.append(f"| {f_} | " + " | ".join(fmt(r[c]) for c in ["audio", "video", "fusion"]
                                                   if c in p) + " |")
        for m in ["audio", "video", "fusion"]:
            if m in p:
                lines.append(f"| **mean** | " if m == "audio" else "")
        lines = [l for l in lines if l != "| **mean** | " and l != ""] + [""]
        means = " | ".join(fmt(p[c].mean()) for c in ["audio", "video", "fusion"] if c in p)
        sds = " | ".join(fmt(p[c].std()) for c in ["audio", "video", "fusion"] if c in p)
        lines.insert(-1, f"| **mean** | {means} |")
        lines.insert(-1, f"| *sd* | {sds} |")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def honest_table(d):
    if d is None:
        return "_actor-independent run not yet complete._"
    g = d.groupby("model")["accuracy"].agg(["mean", "std"])
    rows = ["| model | actor-independent | sd | vs paper's 96.06 |", "|---|---|---|---|"]
    order = ["audio", "video", "fusion-paper", "fusion-oof"]
    names = {"audio": "audio", "video": "video",
             "fusion-paper": "**fusion** (paper's fusion protocol)",
             "fusion-oof": "**fusion** (out-of-fold features)"}
    for m in order:
        if m in g.index:
            rows.append(f"| {names[m]} | **{fmt(g.loc[m,'mean'])}** | {fmt(g.loc[m,'std'])} | "
                        f"{fmt(g.loc[m,'mean'] - PAPER['fusion'])} |")
    extra = ""
    if "fusion-paper" in g.index and "fusion-oof" in g.index:
        fp, fo = g.loc["fusion-paper", "mean"], g.loc["fusion-oof", "mean"]
        extra = (f"\n\nHolding out actors costs **{PAPER['fusion'] - fp:.2f}** points. "
                 f"The fusion-feature leak costs a further **{fp - fo:.2f}**, measured on identical "
                 f"held-out actors, so the two are separable and additive.")
    return "\n".join(rows) + extra


def inject(md, marker, content):
    pat = re.compile(rf"(<!-- AUTO:{marker} -->).*?(<!-- /AUTO:{marker} -->)", re.S)
    if not pat.search(md):
        print(f"  WARNING: marker AUTO:{marker} not found in FINDINGS.md - table not injected")
        return md
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", md)


def main():
    local = "--local" in sys.argv
    if not local:
        print("pulling finished arms from Kaggle (CSVs only):")
        for kernel, subdir, _, _, _ in ARMS:
            pull(kernel, subdir)
        pull(HONEST[0], HONEST[1])
    else:
        print("--local: using results/ as-is")

    arms = []
    for _, subdir, csv, label, changed in ARMS:
        d = load(subdir, csv)
        if d is not None and len(d):
            arms.append((label, changed, d))
    print(f"\narms with data: {[a[0] for a in arms]}")
    honest, honest_src = load_honest()

    md = open(FINDINGS, encoding="utf-8").read()
    md = inject(md, "HEADLINE", headline_table(arms))
    md = inject(md, "PERFOLD", perfold_table(arms))
    md = inject(md, "HONEST", honest_table(honest))
    open(FINDINGS, "w", encoding="utf-8").write(md)
    print("tables injected into docs/FINDINGS.md")

    sys.path.insert(0, os.path.join(REPO, "src"))
    from make_pdf import render
    p, n = render(FINDINGS, PDF,
                  "Findings: reproduction and re-evaluation of Emotion Unlocked")
    print(f"wrote {p} ({n} pages, {os.path.getsize(p)/1024:.0f} KB)")

    best = max((a[2][a[2].model == "fusion"]["accuracy"].mean() for a in arms), default=0)
    print(f"\nbest fusion: {best:.2f}  vs paper 96.06  -> "
          f"{'*** BEATS THE PAPER ***' if best > PAPER['fusion'] else f'short by {PAPER["fusion"] - best:.2f}'}")

    # Verify the rendered PDF rather than trusting it. Warn, don't crash -- the PDF is already
    # written and a partial report beats no report against a deadline.
    import fitz
    full = "".join(pg.get_text() for pg in fitz.open(PDF))
    missing = [x for x in ["96.06", "92.62", "children())[:-1]", "majority baseline"]
               if x not in full]
    if missing:
        print(f"  WARNING: PDF is missing {missing} - check the AUTO markers")
    else:
        print("  PDF verified by text re-extraction.")
    if honest is None:
        print("  NOTE: actor-independent table is a placeholder; re-run when that arm lands.")


if __name__ == "__main__":
    main()
