"""Build an isolated HTRU2 study using the validated, frozen method sources."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BUNDLE = ROOT / "htru2_math_20reps_20260914"
SOURCE = REPO / "real_data_spambase/spambase_c005_20reps_20260914"

def main():
    BUNDLE.mkdir(exist_ok=False)
    for name in ("Ada-PU", "phase3_confirmatory_S001_S040_100reps", "phase5_two_sample_scar_confirmatory_100reps", "cluster"):
        shutil.copytree(SOURCE / name, BUNDLE / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code = BUNDLE / "real_data_htru2"
    code.mkdir()
    for name in ("run.py", "verify_results.py"):
        text = (REPO / "real_data_spambase" / name).read_text(encoding="utf-8")
        text = text.replace("Spambase", "HTRU2").replace("real_data_spambase", "real_data_htru2")
        text = text.replace("p=57,", 'p=splits["train"]["X"].shape[1],')
        (code / name).write_text(text, encoding="utf-8")
    (code / "ranking_repair").mkdir()
    shutil.copy2(REPO / "real_data_spambase/ranking_repair/correction.py", code / "ranking_repair/correction.py")
    # Reuse the auditable split construction with independent, split-specific
    # random streams and prefix sampling to ensure nesting across budgets.
    original = (REPO / "real_data_spambase/data.py").read_text(encoding="utf-8")
    construction = original[original.index("def construct_data"):]
    construction = construction.replace("load_spambase()", "load_htru2()")
    construction = construction.replace('        s = np.zeros(len(background), dtype=int)', '        rng = np.random.default_rng(seed + (10 if name == "train" else 20))\n        s = np.zeros(len(background), dtype=int)')
    construction = construction.replace("rng.binomial(1, c, len(positive_positions))", "(rng.random(len(positive_positions)) < c).astype(int)")
    construction = construction.replace("rng.choice(positive_source, n_p, replace=False)", "rng.permutation(positive_source)[:n_p]")
    construction = construction.replace("rng.choice(background, len(background) - n_p, replace=False)", "rng.permutation(background)[:len(background) - n_p]")
    construction = construction.replace('"sampling_scheme": scheme,', '"dataset": "HTRU2", "positive_sets_nested": True, "sampling_scheme": scheme,')
    (code / "data.py").write_text((ROOT / "loader_header.py").read_text() + "\n\n" + construction, encoding="utf-8")
    shutil.copy2(ROOT / "preflight.py", code / "preflight.py")
    shutil.copy2(ROOT / "cluster.py", BUNDLE / "cluster/cluster.py")
    for path in (BUNDLE / "cluster").iterdir():
        if path.suffix not in (".sh", ".slurm"):
            continue
        text = path.read_text().replace("spam005", "htru2").replace("spambase_", "htru2_")
        text = text.replace("1-40%10", "1-80%10").replace("1-2%2", "1-4%4").replace("3-40%10", "5-80%10")
        text = text.replace("02:00:00", "04:00:00")
        path.write_text(text, encoding="utf-8", newline="\n")

if __name__ == "__main__":
    main()
