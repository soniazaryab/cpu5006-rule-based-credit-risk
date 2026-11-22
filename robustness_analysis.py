"""
Robustness checks for credit_risk_rules.py.
Repeated cross-validation, extra System B variants, statistical tests, costs and age.

Dataset: https://www.kaggle.com/datasets/upadorprofzs/credit-risk
Run: python robustness_analysis.py  (needs pandas, numpy, scipy, scikit-learn, matplotlib)
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)

# Settings (same values as credit_risk_rules.py so results are comparable)
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_BANDS = 4
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50
MAX_ITEMS = 2
A_LOAN_TO_INCOME = 0.10      # System A as first written (thresholds set by eye)
A_AGE = 41.5
FEATURES = ["income", "age", "loan", "loan_to_income"]
HYBRID_FEATURES = ["age", "loan_to_income"]          # features chosen by the expert
NO_AGE_FEATURES = ["income", "loan", "loan_to_income"]

CV_FOLDS = 5
CV_REPEATS = 4               # 5 folds x 4 repeats = 20 evaluations
MDLP_DEPTH = 3               # at most 2**3 = 8 bands per feature
TUNE_BANDS = [3, 4, 5, 6]    # grid searched inside each training fold (System B tuned)
TUNE_SUPPORT = [0.01, 0.02]
TUNE_CONF = [0.5, 0.6, 0.7]
COST_RATIOS = np.arange(1, 21)   # cost of a missed defaulter / cost of a refused good client
AGE_GROUPS = [("under 30", 0, 30), ("30 to 41", 30, 41.5), ("41.5 and over", 41.5, 200)]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "original.csv")
RESULTS_FILE = os.path.join(HERE, "robustness_results.txt")
SCORES_FILE = os.path.join(HERE, "robustness_cv_scores.csv")
CHART_FILE = os.path.join(HERE, "robustness_chart.png")
COST_FILE = os.path.join(HERE, "cost_curve.png")

output_lines = []


def say(text=""):
    print(text)
    output_lines.append(str(text))


def heading(text):
    say("\n" + "=" * 70)
    say(text)
    say("=" * 70)


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


# System A: the expert rule
def predict_a_fixed(data):
    """First version: thresholds chosen by eye from the original training split."""
    return ((data["loan_to_income"] > A_LOAN_TO_INCOME) & (data["age"] < A_AGE)).astype(int).values


def fit_expert(train, use_age=True):
    """Expert procedure written as code, so it can be repeated inside every fold:
    'flag anyone whose loan-to-income is at least the lowest seen among past
    defaulters and (if age is used) who is no older than the oldest past defaulter'."""
    defaulters = train[train["default"] == 1]
    return {"lti": float(defaulters["loan_to_income"].min()),
            "age": float(defaulters["age"].max()) if use_age else None}


def predict_expert(rule, data):
    pred = data["loan_to_income"] >= rule["lti"]
    if rule["age"] is not None:
        pred = pred & (data["age"] <= rule["age"])
    return pred.astype(int).values


# System B: mined class association rules
def _entropy(n1, n):
    """Binary entropy of a group with n1 defaulters out of n (vectorised)."""
    n1, n = np.asarray(n1, float), np.asarray(n, float)
    p = np.divide(n1, n, out=np.zeros_like(n1), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(e)


# AI assistance: this function was drafted with Claude (Anthropic, 2026).
def mdlp_edges(x, y, max_depth=MDLP_DEPTH):
    """Class-guided cut points (Fayyad and Irani, 1993): keep splitting a feature
    where the class labels separate, and stop when the information gain no longer
    pays for the extra cut (minimum description length test)."""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    cuts = []

    def split(lo, hi, depth):
        n = hi - lo
        if depth == 0 or n < 2:
            return
        xx, yy = xs[lo:hi], ys[lo:hi]
        cum = np.cumsum(yy)
        total = cum[-1]
        ent = float(_entropy(total, n))
        if ent == 0:
            return
        idx = np.nonzero(xx[1:] != xx[:-1])[0] + 1       # possible cut positions
        if len(idx) == 0:
            return
        left_n, left_1 = idx, cum[idx - 1]
        right_n, right_1 = n - idx, total - left_1
        split_ent = (left_n * _entropy(left_1, left_n)
                     + right_n * _entropy(right_1, right_n)) / n
        j = int(np.argmin(split_ent))
        gain = ent - split_ent[j]
        k1 = 2 if 0 < left_1[j] < left_n[j] else 1
        k2 = 2 if 0 < right_1[j] < right_n[j] else 1
        delta = np.log2(3 ** 2 - 2) - (2 * ent - k1 * _entropy(left_1[j], left_n[j])
                                       - k2 * _entropy(right_1[j], right_n[j]))
        if gain > (np.log2(n - 1) + delta) / n:
            i = int(idx[j])
            cuts.append(float((xx[i - 1] + xx[i]) / 2))
            split(lo, lo + i, depth - 1)
            split(lo + i, hi, depth - 1)

    split(0, len(xs), max_depth)
    return np.array([-np.inf] + sorted(cuts) + [np.inf])


def make_edges(train, features, method, n_bands):
    edges_by_col = {}
    for col in features:
        if method == "mdlp":
            edges_by_col[col] = mdlp_edges(train[col].values, train["default"].values)
        else:                                            # equal-sized bands (quantiles)
            _, edges = pd.qcut(train[col], n_bands, retbins=True, duplicates="drop")
            edges[0], edges[-1] = -np.inf, np.inf
            edges_by_col[col] = edges
    return edges_by_col


def band_matrix(data, edges_by_col):
    """One True/False column per band, e.g. 'age=band2'. Returns (matrix, names)."""
    cols, names = [], []
    for col, edges in edges_by_col.items():
        idx = pd.cut(data[col], bins=edges, labels=False).values
        for b in range(len(edges) - 1):
            cols.append(idx == b)
            names.append(f"{col}=band{b + 1}")
    return np.column_stack(cols), names


# AI assistance: this function was drafted with Claude (Anthropic, 2026).
def mine_rules(train, features=FEATURES, method="equal", n_bands=N_BANDS,
               min_support=MIN_SUPPORT, min_conf=MIN_CONFIDENCE, max_items=MAX_ITEMS):
    """Mine IF (1-2 band conditions) THEN default rules. Same logic as
    credit_risk_rules.py, written with numpy so it is fast enough to repeat."""
    edges = make_edges(train, features, method, n_bands)
    X, names = band_matrix(train, edges)
    y = train["default"].values
    min_count = min_support * len(train)
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
                rules.append({"combo": combo, "covered": n_cov, "confidence": n_def / n_cov})
    return {"edges": edges, "names": names, "rules": rules}


def predict_rules(model, data):
    X, _ = band_matrix(data, model["edges"])
    pred = np.zeros(len(data), dtype=int)
    for r in model["rules"]:
        pred |= X[:, list(r["combo"])].all(axis=1).astype(int)
    return pred


def describe_band(col, edges, b):
    lo, hi = edges[b], edges[b + 1]
    if np.isinf(lo):
        return f"{col} <= {hi:.3f}"
    if np.isinf(hi):
        return f"{col} > {lo:.3f}"
    return f"{lo:.3f} < {col} <= {hi:.3f}"


def describe_rules(model):
    lines = []
    ranges, k = {}, 0
    for col, edges in model["edges"].items():          # map column index -> text
        for b in range(len(edges) - 1):
            ranges[k] = describe_band(col, edges, b)
            k += 1
    for r in sorted(model["rules"], key=lambda r: -r["confidence"]):
        cond = " AND ".join(ranges[i] for i in r["combo"])
        lines.append(f"  IF {cond} THEN default  "
                     f"(confidence {r['confidence']:.2f}, covers {r['covered']})")
    return lines


# AI assistance: this function was drafted with Claude (Anthropic, 2026).
def tune_b(train):
    """Nested tuning: pick System B's number of bands, support and confidence by
    3-fold cross-validation INSIDE the training data only, so the outer test
    folds are never used to choose settings."""
    inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)
    splits = list(inner.split(train, train["default"]))
    best = None
    for nb, ms, mc in itertools.product(TUNE_BANDS, TUNE_SUPPORT, TUNE_CONF):
        f1s = []
        for a, b in splits:
            m = mine_rules(train.iloc[a], n_bands=nb, min_support=ms, min_conf=mc)
            held = train.iloc[b]
            f1s.append(f1_score(held["default"], predict_rules(m, held), zero_division=0))
        score = float(np.mean(f1s))
        if best is None or score > best[0] + 1e-12:
            best = (score, {"n_bands": nb, "min_support": ms, "min_conf": mc})
    return best[1]


# System C: best single condition (a baseline)
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


# One function per system: learn from `train`, predict `test`
def run_a_fixed(train, test):
    return predict_a_fixed(test), {}


def run_a(train, test):
    rule = fit_expert(train)
    return predict_expert(rule, test), {"lti": rule["lti"], "age": rule["age"]}


def run_a_noage(train, test):
    rule = fit_expert(train, use_age=False)
    return predict_expert(rule, test), {}


def run_b(train, test):
    m = mine_rules(train)
    return predict_rules(m, test), {"model": m}


def run_b_mdlp(train, test):
    m = mine_rules(train, method="mdlp")
    return predict_rules(m, test), {"model": m}


def run_b_tuned(train, test):
    setting = tune_b(train)
    m = mine_rules(train, **setting)
    return predict_rules(m, test), {"model": m, "setting": setting}


def run_hybrid(train, test):
    m = mine_rules(train, features=HYBRID_FEATURES, method="mdlp")
    return predict_rules(m, test), {"model": m}


def run_b_noage(train, test):
    m = mine_rules(train, features=NO_AGE_FEATURES)
    return predict_rules(m, test), {"model": m}


def run_c(train, test):
    rule = fit_single_rule(train)
    return predict_single(rule, test), {"rule": rule}


SYSTEMS = {
    "A_fixed": ("A: expert, thresholds set by eye", run_a_fixed),
    "A": ("A: expert, re-derived in each fold", run_a),
    "B": ("B: mined, equal bands", run_b),
    "B_mdlp": ("B: mined, class-guided bands", run_b_mdlp),
    "B_tuned": ("B: mined, equal bands, tuned", run_b_tuned),
    "H": ("H: hybrid (expert features, mined thresholds)", run_hybrid),
    "C": ("C: best single rule", run_c),
    "A_noage": ("A without age", run_a_noage),
    "B_noage": ("B without age", run_b_noage),
}


def scores(actual, pred):
    return {"precision": precision_score(actual, pred, zero_division=0),
            "recall": recall_score(actual, pred, zero_division=0),
            "f1": f1_score(actual, pred, zero_division=0)}


def corrected_t_test(diff, n_train, n_test):
    """Nadeau and Bengio (2003): a t-test for cross-validation runs whose training
    sets overlap. The usual 1/J variance is enlarged by n_test / n_train."""
    j = len(diff)
    mean, var = diff.mean(), diff.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var)
    if se == 0:
        return mean, mean, mean, 0.0 if mean != 0 else 1.0
    t = mean / se
    half = stats.t.ppf(0.975, j - 1) * se
    return mean, mean - half, mean + half, 2 * stats.t.sf(abs(t), j - 1)


# 1. Repeated stratified cross-validation of every system
heading(f"1. Repeated stratified {CV_FOLDS}-fold cross-validation x {CV_REPEATS} repeats")
say("Every system learns only from the training folds of each evaluation.")
rskf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                               random_state=RANDOM_SEED)
rows, oof, fold_info = [], {k: np.full(len(df), -1) for k in SYSTEMS}, {k: [] for k in SYSTEMS}
n_train_sizes, n_test_sizes = [], []
for k, (tr, te) in enumerate(rskf.split(df, df["default"])):
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    n_train_sizes.append(len(train))
    n_test_sizes.append(len(test))
    y = test["default"].values
    for key, (label, fn) in SYSTEMS.items():
        pred, info = fn(train, test)
        fold_info[key].append(info)
        if k < CV_FOLDS:                                 # first repeat covers every client once
            oof[key][te] = pred
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn_ = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({"evaluation": k + 1, "system": key, **scores(y, pred),
                     "tp": tp, "fp": fp, "fn": fn_, "tn": tn})
    print(f"  evaluation {k + 1}/{CV_FOLDS * CV_REPEATS} done")

cv = pd.DataFrame(rows)
cv.to_csv(SCORES_FILE, index=False)
summary = cv.groupby("system")[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
summary = summary.loc[list(SYSTEMS)]
say("\nMean and standard deviation over the 20 evaluations:")
say(summary.to_string())
say("\nSystem keys: " + "; ".join(f"{k} = {v[0]}" for k, v in SYSTEMS.items()))

a_th = pd.DataFrame([i for i in fold_info["A"]])
say(f"\nSystem A thresholds re-derived in each fold: loan-to-income >= "
    f"{a_th.lti.mean():.3f} (range {a_th.lti.min():.3f} to {a_th.lti.max():.3f}); "
    f"age <= {a_th.age.mean():.1f} (range {a_th.age.min():.1f} to {a_th.age.max():.1f})")
tuned = pd.DataFrame([i["setting"] for i in fold_info["B_tuned"]])
say("Settings chosen by the inner loop for tuned System B (number of folds picking each):")
for col in tuned.columns:
    say(f"  {col}: " + ", ".join(f"{v} x{c}" for v, c in tuned[col].value_counts().sort_index().items()))
n_rules = {k: np.mean([len(i["model"]["rules"]) for i in fold_info[k]])
           for k in ["B", "B_mdlp", "B_tuned", "H", "B_noage"]}
say("Average number of mined rules per fold: " + ", ".join(f"{k} {v:.1f}" for k, v in n_rules.items()))

# 2. Paired comparisons with the corrected resampled t-test
heading("2. Paired comparisons of F1 (corrected resampled t-test, Nadeau and Bengio, 2003)")
say("Each line: first system minus second, one difference per evaluation (n = 20).")
say("A plain paired t-test would treat the 20 evaluations as independent; they are not,")
say("because the training sets overlap, so the corrected test is used instead.")
n_tr, n_te = float(np.mean(n_train_sizes)), float(np.mean(n_test_sizes))
f1_of = {k: cv[cv.system == k].sort_values("evaluation")["f1"].values for k in SYSTEMS}
pairs = [("A", "B"), ("B_mdlp", "B"), ("B_tuned", "B"), ("A", "B_mdlp"), ("A", "B_tuned"),
         ("B_mdlp", "B_tuned"), ("H", "A"),
         ("A_fixed", "A"), ("A", "C"), ("B", "C")]
for x, y_ in pairs:
    d = f1_of[x] - f1_of[y_]
    mean, lo, hi, p = corrected_t_test(d, n_tr, n_te)
    say(f"  {x:8s} - {y_:8s}: mean diff {mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
        f"p = {p:.3f}; {x} higher in {int((d > 0).sum())}, {y_} higher in {int((d < 0).sum())}, "
        f"ties {int((d == 0).sum())}")

# 3. Original split (same as credit_risk_rules.py: seed 42, stratified 80/20)
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                               stratify=df["default"])
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
y_test = test["default"].values
heading("3. Original train/test split: every system")
split_preds, split_info = {}, {}
split_rows = []
for key, (label, fn) in SYSTEMS.items():
    pred, info = fn(train, test)
    split_preds[key], split_info[key] = pred, info
    s = scores(y_test, pred)
    split_rows.append({"system": key, "flagged": int(pred.sum()),
                       "TP": int(((pred == 1) & (y_test == 1)).sum()),
                       "FP": int(((pred == 1) & (y_test == 0)).sum()),
                       "FN": int(((pred == 0) & (y_test == 1)).sum()),
                       "precision": round(s["precision"], 3),
                       "recall": round(s["recall"], 3), "F1": round(s["f1"], 3)})
say(pd.DataFrame(split_rows).to_string(index=False))

rule_a = fit_expert(train)
say(f"\nSystem A re-derived on this training split: IF loan_to_income >= {rule_a['lti']:.3f} "
    f"AND age <= {rule_a['age']:.1f} THEN default")
rule_c = split_info["C"]["rule"]
say(f"System C: IF {rule_c['col']} {rule_c['dir']} {rule_c['t']:.3f} THEN default "
    f"(training F1 {rule_c['f1']:.3f})")
for key in ["B", "B_mdlp", "H"]:
    model = split_info[key]["model"]
    say(f"\nRules mined for {key} ({len(model['rules'])} rules):")
    for line in describe_rules(model):
        say(line)
    for col, edges in model["edges"].items():
        cuts = ", ".join(f"{e:.3f}" for e in edges[1:-1]) or "none"
        say(f"  cut points for {col}: {cuts}")
say(f"\nTuned System B chose: {split_info['B_tuned']['setting']}")

say("\nMcNemar's exact test on the original test split (do the systems err on different clients?)")
for x, y_ in [("A", "B"), ("B_mdlp", "B"), ("B_tuned", "B"), ("H", "A")]:
    wrong_x, wrong_y = split_preds[x] != y_test, split_preds[y_] != y_test
    b, c = int((wrong_x & ~wrong_y).sum()), int((~wrong_x & wrong_y).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    say(f"  {x} vs {y_}: {x} alone wrong on {b} clients, {y_} alone wrong on {c}; p = {p:.3f}")

# 4. Sensitivity analysis
heading("4a. Sensitivity of System B (5-fold CV on the TRAINING split only)")
say("Mean F1 over 5 folds; 'rules' = rules mined on the full training split.")
say("(F1 of 0 or close to 0 means too few or no rules were found, so nothing was flagged.)")
grid_rows = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_SEED)
splits = list(skf.split(train, train["default"]))
for nb, ms, mc in itertools.product([3, 4, 5, 6], [0.005, 0.01, 0.02], [0.4, 0.5, 0.6, 0.7]):
    f1s = []
    for tr_i, te_i in splits:
        m = mine_rules(train.iloc[tr_i], n_bands=nb, min_support=ms, min_conf=mc)
        t = train.iloc[te_i]
        f1s.append(f1_score(t["default"], predict_rules(m, t), zero_division=0))
    n_r = len(mine_rules(train, n_bands=nb, min_support=ms, min_conf=mc)["rules"])
    grid_rows.append({"bands": nb, "min_support": ms, "min_confidence": mc,
                      "rules": n_r, "mean_F1": round(float(np.mean(f1s)), 3)})
grid = pd.DataFrame(grid_rows)
say(grid.to_string(index=False))
say(f"\nF1 ranges from {grid.mean_F1.min():.3f} to {grid.mean_F1.max():.3f} "
    f"across {len(grid)} settings (default setting: 4 bands, 0.01, 0.5).")

heading("4b. Sensitivity of System A thresholds set by eye (evaluated on the TRAINING split)")
a_rows = []
for lti, age in itertools.product([0.08, 0.10, 0.12, 0.15], [35.0, 41.5, 45.0, 50.0]):
    pred = ((train["loan_to_income"] > lti) & (train["age"] < age)).astype(int).values
    s = scores(train["default"].values, pred)
    a_rows.append({"loan_to_income >": lti, "age <": age, "precision": round(s["precision"], 3),
                   "recall": round(s["recall"], 3), "F1": round(s["f1"], 3)})
say(pd.DataFrame(a_rows).to_string(index=False))

# 5. Error analysis on the original test split (System A vs System B)
heading("5. Error analysis (original test split, System A re-derived vs System B)")
test = test.copy()
test["pred_A"], test["pred_B"] = split_preds["A"], split_preds["B"]
groups = {
    "Defaulters caught by BOTH": test[(test.default == 1) & (test.pred_A == 1) & (test.pred_B == 1)],
    "Defaulters caught by A only (missed by B)": test[(test.default == 1) & (test.pred_A == 1) & (test.pred_B == 0)],
    "Defaulters caught by B only (missed by A)": test[(test.default == 1) & (test.pred_A == 0) & (test.pred_B == 1)],
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
    age_cut = float(split_info["B"]["model"]["edges"]["age"][1])
    say(f"  Of these, {int((missed['age'] > age_cut).sum())} are older than {age_cut:.2f} "
        f"(the top of the youngest equal-sized age band).")

# 6. Cost analysis: when is each system cheaper?
heading("6. Expected cost per client (pooled over the 20 evaluations)")
say("cost = (r x missed defaulters + refused good clients) / clients, where r is the cost of")
say("one missed defaulter relative to one refused good client (a refused good client = 1).")
pool = cv.groupby("system")[["tp", "fp", "fn", "tn"]].sum()
n_pool = int(pool.iloc[0].sum())
n_pos = int(pool.loc["A", ["tp", "fn"]].sum())


def cost(key, r):
    return (r * pool.loc[key, "fn"] + pool.loc[key, "fp"]) / n_pool


def break_even(x, y_):
    """Ratio r above which x is cheaper than y_ (x makes fewer misses, more refusals)."""
    d_fn = pool.loc[y_, "fn"] - pool.loc[x, "fn"]
    d_fp = pool.loc[x, "fp"] - pool.loc[y_, "fp"]
    if d_fn > 0 and d_fp > 0:
        return d_fp / d_fn, f"{x} is cheaper when a missed defaulter costs more than this many refused clients"
    if d_fn < 0 and d_fp < 0:
        return d_fp / d_fn, f"{y_} is cheaper when a missed defaulter costs more than this many refused clients"
    return None, f"no break-even: one of the two is better on both kinds of error (or equal)"


cost_rows = []
for key in ["A", "B", "B_mdlp", "B_tuned", "H", "C"]:
    cost_rows.append({"system": key, "misses": int(pool.loc[key, "fn"]),
                      "refusals": int(pool.loc[key, "fp"]),
                      **{f"cost r={r}": round(cost(key, r), 3) for r in (1, 5, 10, 20)}})
cost_rows.append({"system": "flag nobody", "misses": n_pos, "refusals": 0,
                  **{f"cost r={r}": round(r * n_pos / n_pool, 3) for r in (1, 5, 10, 20)}})
say(pd.DataFrame(cost_rows).to_string(index=False))
say("\nBreak-even ratios:")
for x, y_ in [("A", "B"), ("A", "B_mdlp"), ("A", "B_tuned"), ("B_mdlp", "B"), ("B_tuned", "B")]:
    r_star, note = break_even(x, y_)
    say(f"  {x} vs {y_}: " + (f"r* = {r_star:.2f}; {note}" if r_star is not None else note))

# 7. Age: what does each system lose without it, and who does it flag?
heading("7a. Effect of removing age (mean over the 20 evaluations)")
abl = summary[["precision", "recall", "f1"]].xs("mean", axis=1, level=1)
say(abl.loc[["A", "A_noage", "B", "B_noage"]].to_string())
for with_age, without in [("A", "A_noage"), ("B", "B_noage")]:
    d = f1_of[with_age] - f1_of[without]
    mean, lo, hi, p = corrected_t_test(d, n_tr, n_te)
    say(f"  F1 lost by removing age from {with_age}: {mean:+.3f} "
        f"(95% CI [{lo:+.3f}, {hi:+.3f}], p = {p:.3f})")

heading("7b. Share of each age group flagged (out-of-fold predictions, first repeat)")
age_rows = []
for name, lo, hi in AGE_GROUPS:
    in_group = (df["age"] >= lo) & (df["age"] < hi)
    row = {"age group": name, "clients": int(in_group.sum()),
           "defaulted %": round(100 * df.loc[in_group, "default"].mean(), 1)}
    for key in ["A", "B", "A_noage", "B_noage"]:
        row[f"{key} flagged %"] = round(100 * float(np.mean(oof[key][in_group.values])), 1)
    age_rows.append(row)
say(pd.DataFrame(age_rows).to_string(index=False))

# 8. Charts (PNG and PDF)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["A", "B", "B_mdlp", "B_tuned", "H", "C"]
    short = {"A": "A\nexpert", "B": "B\nequal", "B_mdlp": "B\nclass-guided",
             "B_tuned": "B\ntuned", "H": "H\nhybrid", "C": "C\nsingle"}
    g = cv.groupby("system")["f1"].agg(["mean", "std"]).loc[show]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([short[k] for k in show], g["mean"], yerr=g["std"], capsize=4,
           color=["#4C72B0", "#DD8452", "#C44E52", "#8172B3", "#55A868", "#937860"])
    ax.set_ylabel("F1 (mean and SD, 20 evaluations)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=300)
    fig.savefig(CHART_FILE.replace(".png", ".pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, colour, style in [("A", "#4C72B0", "-"), ("B_mdlp", "#C44E52", "-"),
                               ("B_tuned", "#8172B3", "-")]:
        ax.plot(COST_RATIOS, [cost(key, r) for r in COST_RATIOS], style, label=SYSTEMS[key][0],
                color=colour)
    for other in ["B_mdlp", "B_tuned"]:
        r_star, _ = break_even("A", other)
        if r_star is not None and COST_RATIOS.min() <= r_star <= COST_RATIOS.max():
            ax.axvline(r_star, ls=":", color="grey")
            ax.text(r_star + 0.2, ax.get_ylim()[1] * 0.9, f"r = {r_star:.1f}", fontsize=8)
    ax.set_xlabel("Cost of a missed defaulter relative to a refused good client (r)")
    ax.set_ylabel("Expected cost per client")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(COST_FILE, dpi=300)
    fig.savefig(COST_FILE.replace(".png", ".pdf"))
    plt.close(fig)
    say(f"\nSaved charts: {CHART_FILE} and {COST_FILE} (and .pdf copies)")
except ImportError:
    say("\n(matplotlib not installed, so no charts were drawn: pip install matplotlib)")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"\nSaved: {RESULTS_FILE} and {SCORES_FILE}")