"""
CPU5006 Assessment 1: Rule-based AI on the Kaggle "Credit risk" dataset.

Research question:
    Which rule-based approach more effectively identifies customers who will
    default on financing: an expert-written rule set (System A), or a rule set
    mined automatically from the data (System B)?

Dataset: Murilao (2020) "Credit risk" [Dataset]. Kaggle.
         https://www.kaggle.com/datasets/upadorprofzs/credit-risk
"""

import itertools
import os

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

# Settings (all design decisions in one place so they can be reported)
RANDOM_SEED = 42          # makes the train/test split repeatable
TEST_SIZE = 0.20          # 80% train, 20% test
N_BANDS = 4               # System B: each number is sorted into 4 equal-sized bands
MIN_SUPPORT = 0.01        # System B: a rule must cover at least 1% of training rows
MIN_CONFIDENCE = 0.50     # System B: a rule must be right at least 50% of the time
MAX_ITEMS = 2             # System B: rules use at most 2 conditions (keeps them readable)

# System A thresholds, chosen by looking at the TRAINING data only
A_LOAN_TO_INCOME = 0.10
A_AGE = 41.5

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "original.csv")
RESULTS_FILE = os.path.join(HERE, "results.txt")
RULES_FILE = os.path.join(HERE, "mined_rules.csv")

output_lines = []


def say(text=""):
    """Print and also keep the text so it can be saved to results.txt."""
    print(text)
    output_lines.append(str(text))


# 1. Load and clean the data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    # Fallback: download from Kaggle if the local copy is missing
    import kagglehub
    path = kagglehub.dataset_download("upadorprofzs/credit-risk")
    return pd.read_csv(os.path.join(path, "original.csv"))


df = load_data()
say(f"Rows at start: {len(df)}")
say(f"Missing ages: {int(df['age'].isnull().sum())}")
say(f"Negative ages: {int((df['age'] < 0).sum())}")

df = df.drop(columns=["clientid"])                      # an ID, not useful for rules
df = df[df["age"].notnull() & (df["age"] >= 18)]        # remove missing / impossible ages
df = df.reset_index(drop=True)
df["loan_to_income"] = df["loan"] / df["income"]        # engineered feature
say(f"Rows after cleaning: {len(df)}")

train, test = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["default"]
)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
say(f"Train rows: {len(train)} | Test rows: {len(test)}")
say(f"Default rate in train: {train['default'].mean():.4f} | test: {test['default'].mean():.4f}")


# 2. System A: expert-written rule
def system_a(data):
    """IF loan-to-income > 0.10 AND age < 41.5 THEN default ELSE paid."""
    rule = (data["loan_to_income"] > A_LOAN_TO_INCOME) & (data["age"] < A_AGE)
    return rule.astype(int)


# 3. System B: class association rules mined from the training data
FEATURES = ["income", "age", "loan", "loan_to_income"]

# Band edges come from the TRAINING data only, then are reused on the test data
band_edges = {}
for col in FEATURES:
    _, edges = pd.qcut(train[col], N_BANDS, retbins=True, duplicates="drop")
    edges[0], edges[-1] = float("-inf"), float("inf")   # so unseen values still fit a band
    band_edges[col] = edges


def to_bands(data):
    """Turn each number into a band label such as 'age=band2'."""
    out = pd.DataFrame(index=data.index)
    for col in FEATURES:
        labels = [f"{col}=band{i + 1}" for i in range(len(band_edges[col]) - 1)]
        out[col] = pd.cut(data[col], bins=band_edges[col], labels=labels)
    return out.astype(str)


def describe_band(item):
    col, band = item.split("=")
    i = int(band.replace("band", "")) - 1
    lo, hi = band_edges[col][i], band_edges[col][i + 1]
    if lo == float("-inf"):
        return f"{col} <= {hi:.3f}"
    if hi == float("inf"):
        return f"{col} > {lo:.3f}"
    return f"{lo:.3f} < {col} <= {hi:.3f}"


def mine_rules(train_bands, labels):
    """Find IF (conditions) THEN default rules that meet support and confidence."""
    n = len(train_bands)
    min_count = MIN_SUPPORT * n
    # Each row becomes a set of items, e.g. {'age=band1', 'loan=band4'}
    rows = [set(r) for r in train_bands.values.tolist()]
    items = sorted({it for r in rows for it in r})
    rules = []
    for size in range(1, MAX_ITEMS + 1):
        for combo in itertools.combinations(items, size):
            # one condition per feature only
            if len({c.split("=")[0] for c in combo}) < size:
                continue
            combo_set = set(combo)
            covered = [i for i, r in enumerate(rows) if combo_set <= r]
            if len(covered) < min_count:
                continue
            defaults = int(labels.iloc[covered].sum())
            confidence = defaults / len(covered)
            if confidence >= MIN_CONFIDENCE and defaults >= min_count:
                rules.append({
                    "conditions": combo,
                    "covered": len(covered),
                    "defaults": defaults,
                    "support": round(len(covered) / n, 4),
                    "confidence": round(confidence, 4),
                })
    # Strongest rules first
    rules.sort(key=lambda r: (-r["confidence"], -r["covered"]))
    return rules


train_bands = to_bands(train)
rules = mine_rules(train_bands, train["default"])


def system_b(data):
    """Predict default if ANY mined rule fires, otherwise predict paid."""
    bands = to_bands(data)
    rows = [set(r) for r in bands.values.tolist()]
    preds = []
    for r in rows:
        preds.append(int(any(set(rule["conditions"]) <= r for rule in rules)))
    return pd.Series(preds, index=data.index)


# 4. Baseline: always predict "paid"
def baseline(data):
    return pd.Series(0, index=data.index)


# 5. Evaluate everything
def evaluate(name, data, predictor):
    actual = data["default"]
    pred = predictor(data)
    tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
    return {
        "System": name,
        "Accuracy": round(accuracy_score(actual, pred), 4),
        "Precision": round(precision_score(actual, pred, zero_division=0), 4),
        "Recall": round(recall_score(actual, pred, zero_division=0), 4),
        "F1": round(f1_score(actual, pred, zero_division=0), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


say("\n" + "=" * 70)
say(f"System A (expert rule): IF loan_to_income > {A_LOAN_TO_INCOME} AND age < {A_AGE} THEN default")
say("=" * 70)

say("\n" + "=" * 70)
say(f"System B (mined rules): {len(rules)} rules found")
say(f"Settings: {N_BANDS} bands, min support {MIN_SUPPORT}, min confidence {MIN_CONFIDENCE}, "
    f"max {MAX_ITEMS} conditions")
say("=" * 70)
for k, r in enumerate(rules[:10], start=1):
    text = " AND ".join(describe_band(c) for c in r["conditions"])
    say(f"{k:>2}. IF {text} THEN default  "
        f"(confidence {r['confidence']:.2f}, covers {r['covered']} rows)")
if len(rules) > 10:
    say(f"    ... plus {len(rules) - 10} more (see mined_rules.csv)")

pd.DataFrame([{
    "rule": " AND ".join(describe_band(c) for c in r["conditions"]),
    "n_conditions": len(r["conditions"]),
    "covered_rows": r["covered"],
    "defaults": r["defaults"],
    "support": r["support"],
    "confidence": r["confidence"],
} for r in rules]).to_csv(RULES_FILE, index=False)

for label, data in [("TRAINING data", train), ("TEST data (final results)", test)]:
    table = pd.DataFrame([
        evaluate("Baseline (always paid)", data, baseline),
        evaluate("System A (expert)", data, system_a),
        evaluate("System B (mined)", data, system_b),
    ])
    say("\n" + "=" * 70)
    say(f"Results on {label} ({len(data)} rows, {int(data['default'].sum())} defaulters)")
    say("=" * 70)
    say(table.to_string(index=False))

say("\nTP = defaulters caught, FP = good customers wrongly flagged,")
say("FN = defaulters missed, TN = good customers correctly passed.")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"\nSaved: {RESULTS_FILE} and {RULES_FILE}")