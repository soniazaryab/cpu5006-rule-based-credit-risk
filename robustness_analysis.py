"""
Robustness checks for credit_risk_rules.py.
Repeated cross-validation, single-rule baseline, sensitivity and error analysis.

Dataset: https://www.kaggle.com/datasets/upadorprofzs/credit-risk
Run: python robustness_analysis.py  (needs pandas, numpy, scipy, scikit-learn, matplotlib)
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     train_test_split)

# Settings (same values as credit_risk_rules.py so results are comparable)

RANDOM_SEED = 42
TEST_SIZE = 0.20
N_BANDS = 4
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50
MAX_ITEMS = 2
A_LOAN_TO_INCOME = 0.10
A_AGE = 41.5
FEATURES = ["income", "age", "loan", "loan_to_income"]

CV_FOLDS = 5
CV_REPEATS = 4            # 5 folds x 4 repeats = 20 evaluations

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "original.csv")
RESULTS_FILE = os.path.join(HERE, "robustness_results.txt")
SCORES_FILE = os.path.join(HERE, "robustness_cv_scores.csv")
CHART_FILE = os.path.join(HERE, "robustness_chart.png")

output_lines = []


def say(text=""):
    print(text)
    output_lines.append(str(text))


# Load and clean (identical steps to credit_risk_rules.py)
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    import kagglehub
    path = kagglehub.dataset_download("upadorprofzs/credit-risk")
    return pd.read_csv(os.path.join(path, "original.csv"))


df = load_data()
df = df.drop(columns=["clientid"])
df = df[df["age"].notnull() & (df["age"] >= 18)].reset_index(drop=True)
df["loan_to_income"] = df["loan"] / df["income"]
say(f"Rows after cleaning: {len(df)} | defaulters: {int(df['default'].sum())}")


# The three systems (all learn from the TRAINING data only)
def system_a(data, lti=A_LOAN_TO_INCOME, age=A_AGE):
    """Expert rule: IF loan-to-income > lti AND age < age THEN default."""
    return ((data["loan_to_income"] > lti) & (data["age"] < age)).astype(int).values


def make_band_edges(train, n_bands):
    edges_by_col = {}
    for col in FEATURES:
        _, edges = pd.qcut(train[col], n_bands, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
        edges_by_col[col] = edges
    return edges_by_col


def band_matrix(data, edges_by_col):
    """One True/False column per band, e.g. 'age=band2'. Returns (matrix, names)."""
    cols, names = [], []
    for col in FEATURES:
        edges = edges_by_col[col]
        idx = pd.cut(data[col], bins=edges, labels=False).values
        for b in range(len(edges) - 1):
            cols.append(idx == b)
            names.append(f"{col}=band{b + 1}")
    return np.column_stack(cols), names


# AI assistance: this function was drafted with Claude (Anthropic, 2026).
def mine_rules(train, n_bands=N_BANDS, min_support=MIN_SUPPORT,
               min_conf=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Same logic as credit_risk_rules.py, written with numpy so it is fast
    enough to repeat many times."""
    edges = make_band_edges(train, n_bands)
    X, names = band_matrix(train, edges)
    y = train["default"].values
    n = len(train)
    min_count = min_support * n
    feature_of = [nm.split("=")[0] for nm in names]
    rules = []
    for size in range(1, max_items + 1):
        for combo in itertools.combinations(range(len(names)), size):
            if len({feature_of[i] for i in combo}) < size:
                continue
            covered = X[:, list(combo)].all(axis=1)
            n_cov = int(covered.sum())
            if n_cov < min_count:
                continue
            n_def = int(y[covered].sum())
            if n_def / n_cov >= min_conf and n_def >= min_count:
                rules.append({"combo": combo, "covered": n_cov,
                              "confidence": n_def / n_cov})
    return {"edges": edges, "names": names, "rules": rules}


def predict_b(model, data):
    X, _ = band_matrix(data, model["edges"])
    pred = np.zeros(len(data), dtype=int)
    for r in model["rules"]:
        pred |= X[:, list(r["combo"])].all(axis=1).astype(int)
    return pred


# AI assistance: this function was drafted with Claude (Anthropic, 2026).
def fit_single_rule(train):
    """Best ONE-condition rule: pick the feature, direction and threshold
    (from 5%, 10% ... 95% quantiles) with the best F1 on the training data."""
    y = train["default"].values
    best = {"f1": -1}
    for col in FEATURES:
        for q in np.arange(0.05, 0.96, 0.05):
            t = float(train[col].quantile(q))
            for direction in (">", "<="):
                pred = (train[col] > t) if direction == ">" else (train[col] <= t)
                f1 = f1_score(y, pred.astype(int), zero_division=0)
                if f1 > best["f1"]:
                    best = {"f1": f1, "col": col, "t": t, "dir": direction}
    return best


def predict_single(rule, data):
    s = data[rule["col"]]
    return ((s > rule["t"]) if rule["dir"] == ">" else (s <= rule["t"])).astype(int).values


def scores(actual, pred):
    return {"precision": precision_score(actual, pred, zero_division=0),
            "recall": recall_score(actual, pred, zero_division=0),
            "f1": f1_score(actual, pred, zero_division=0)}

# 1. Repeated stratified cross-validation
say("\n" + "=" * 70)
say(f"1. Repeated stratified {CV_FOLDS}-fold cross-validation x {CV_REPEATS} repeats")
say("=" * 70)

rskf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                               random_state=RANDOM_SEED)
rows = []
for k, (tr, te) in enumerate(rskf.split(df, df["default"])):
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    y = test["default"].values
    model = mine_rules(train)
    single = fit_single_rule(train)
    for name, pred in [("A: expert rule", system_a(test)),
                       ("B: mined rules", predict_b(model, test)),
                       ("C: best single rule", predict_single(single, test))]:
        rows.append({"evaluation": k + 1, "system": name, **scores(y, pred)})

cv = pd.DataFrame(rows)
cv.to_csv(SCORES_FILE, index=False)
summary = cv.groupby("system")[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
say(summary.to_string())

say("\nPaired comparison of F1 (A minus B), one difference per evaluation:")
f1_a = cv[cv.system == "A: expert rule"]["f1"].values
f1_b = cv[cv.system == "B: mined rules"]["f1"].values
diff = f1_a - f1_b
n = len(diff)
mean_d, sd_d = diff.mean(), diff.std(ddof=1)
half = stats.t.ppf(0.975, n - 1) * sd_d / np.sqrt(n)
say(f"  mean difference = {mean_d:.3f}, 95% CI [{mean_d - half:.3f}, {mean_d + half:.3f}]")
say(f"  System A had the higher F1 in {int((diff > 0).sum())} of {n} evaluations "
    f"(B higher in {int((diff < 0).sum())}, ties {int((diff == 0).sum())})")
say("  Note: the 20 evaluations reuse the same customers, so they are not fully")
say("  independent. Treat the interval as a guide, not an exact significance test.")
say("  Note: System A's thresholds were chosen by looking at the original training")
say("  split, which overlaps these folds, so A is slightly favoured here.")

# Original split (same as credit_risk_rules.py: seed 42, stratified 80/20)
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                               stratify=df["default"])
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
model = mine_rules(train)
single = fit_single_rule(train)

say("\n" + "=" * 70)
say("2. Best single-condition rule (System C), learned on the training split")
say("=" * 70)
say(f"IF {single['col']} {single['dir']} {single['t']:.3f} THEN default "
    f"(training F1 {single['f1']:.3f})")
pc = predict_single(single, test)
sc = scores(test["default"].values, pc)
say(f"Original test split: precision {sc['precision']:.3f}, "
    f"recall {sc['recall']:.3f}, F1 {sc['f1']:.3f}")

# 3. Sensitivity analysis
say("\n" + "=" * 70)
say("3a. Sensitivity of System B (5-fold CV on the TRAINING split only)")
say("=" * 70)
say("Mean F1 over 5 folds; 'rules' = rules mined on the full training split.")
say("(F1 of 0 or close to 0 means too few or no rules were found, so nothing was flagged.)")
grid_rows = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_SEED)
splits = list(skf.split(train, train["default"]))
for nb, ms, mc in itertools.product([3, 4, 5, 6], [0.005, 0.01, 0.02], [0.4, 0.5, 0.6, 0.7]):
    f1s = []
    for tr, te in splits:
        m = mine_rules(train.iloc[tr].reset_index(drop=True), nb, ms, mc)
        t = train.iloc[te].reset_index(drop=True)
        f1s.append(f1_score(t["default"], predict_b(m, t), zero_division=0))
    n_rules = len(mine_rules(train, nb, ms, mc)["rules"])
    grid_rows.append({"bands": nb, "min_support": ms, "min_confidence": mc,
                      "rules": n_rules, "mean_F1": round(float(np.mean(f1s)), 3)})
grid = pd.DataFrame(grid_rows)
say(grid.to_string(index=False))
say(f"\nF1 ranges from {grid.mean_F1.min():.3f} to {grid.mean_F1.max():.3f} "
    f"across {len(grid)} settings (default setting: 4 bands, 0.01, 0.5).")

say("\n" + "=" * 70)
say("3b. Sensitivity of System A thresholds (evaluated on the TRAINING split)")
say("=" * 70)
a_rows = []
for lti, age in itertools.product([0.08, 0.10, 0.12, 0.15], [35.0, 41.5, 45.0, 50.0]):
    s = scores(train["default"].values, system_a(train, lti, age))
    a_rows.append({"loan_to_income >": lti, "age <": age,
                   "precision": round(s["precision"], 3),
                   "recall": round(s["recall"], 3), "F1": round(s["f1"], 3)})
say(pd.DataFrame(a_rows).to_string(index=False))


# 4. Error analysis on the original test split
say("\n" + "=" * 70)
say("4. Error analysis (original test split)")
say("=" * 70)
test = test.copy()
test["pred_A"] = system_a(test)
test["pred_B"] = predict_b(model, test)
groups = {
    "Defaulters caught by BOTH": test[(test.default == 1) & (test.pred_A == 1) & (test.pred_B == 1)],
    "Defaulters caught by A only (missed by B)": test[(test.default == 1) & (test.pred_A == 1) & (test.pred_B == 0)],
    "Defaulters missed by BOTH": test[(test.default == 1) & (test.pred_A == 0) & (test.pred_B == 0)],
    "Good customers wrongly flagged by A": test[(test.default == 0) & (test.pred_A == 1)],
    "Good customers wrongly flagged by B": test[(test.default == 0) & (test.pred_B == 1)],
}
err_rows = []
for label, g in groups.items():
    err_rows.append({"group": label, "n": len(g),
                     "median age": round(g["age"].median(), 1) if len(g) else None,
                     "median income": round(g["income"].median()) if len(g) else None,
                     "median loan": round(g["loan"].median()) if len(g) else None,
                     "median loan/income": round(g["loan_to_income"].median(), 3) if len(g) else None})
say(pd.DataFrame(err_rows).to_string(index=False))

missed = groups["Defaulters caught by A only (missed by B)"]
if len(missed):
    say(f"\nAges of the {len(missed)} defaulters that A caught but B missed:")
    say("  " + ", ".join(f"{a:.0f}" for a in sorted(missed["age"])))
    age_cut = float(model["edges"]["age"][1])   # upper edge of the youngest age band
    say(f"  Of these, {int((missed['age'] > age_cut).sum())} are older than {age_cut:.2f} "
        f"(the youngest age band, which the mined rules rely on if they all need it).")

# 5. Chart
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = cv.groupby("system")["f1"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index, g["mean"], yerr=g["std"], capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("F1 score (mean ± SD over 20 evaluations)")
    ax.set_title("Defaulter detection across repeated cross-validation")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=200)
    say(f"\nSaved chart: {CHART_FILE}")
except ImportError:
    say("\n(matplotlib not installed, so no chart was drawn: pip install matplotlib)")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"\nSaved: {RESULTS_FILE} and {SCORES_FILE}")