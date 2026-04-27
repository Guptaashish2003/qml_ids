"""
QML Intrusion Detection — Full Pipeline
========================================
Steps:
  1. Load IoT-23 conn.log.labeled files
  2. Clean & preprocess (NaN, zero-variance, outlier capping)
  3. Encode features → qubit circuits (demo)
  4. Optimize hyperparameters (C, n_qubits) via 3-fold CV grid search
  5. Train SVM baseline + QSVM (+ QCNN in full mode)
  6. Evaluate: confusion matrix, ROC curves, accuracy summary

Run commands
max_rows_per_capture=100_000,  # With your IoT-23 dataset (recommended):
  python main.py --iot23 /path/to/CTU-IoT-folder

  # Full pipeline + QCNN + training time benchmark:
  python main.py --iot23 /path/to/CTU-IoT-folder --mode full

  # Binary classification (Benign vs Malicious):
  python main.py --iot23 /path/to/CTU-IoT-folder --binary

  # Skip optimizer and use defaults (faster):
  python main.py --iot23 /path/to/CTU-IoT-folder --skip-optimize

  # Synthetic demo (no dataset needed):
  python main.py --demo
"""

import argparse
import sys
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.stream_builder import StreamBuilder
from data.iot23_loader import IoT23Loader
from encoder.qubit_encoder import QuantumEncoder
from models.qsvm_model import QSVMDetector, SVMBaseline
from models.qcnn_model import QCNNDetector, CNNBaseline
from utils.evaluation import Evaluator


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════
def load_iot23(dataset_dir: str, n_samples: int, binary: bool = False):
    print("\n" + "="*60)
    print("  STEP 1: Loading IoT-23 Dataset")
    print("="*60)

    loader = IoT23Loader(
        dataset_dir=dataset_dir,
        binary=binary,
        max_rows_per_capture=100_000,
    )

    captures = loader.list_captures()
    print(f"\nFound {len(captures)} capture folder(s):")
    for c in captures:
        print(f"  • {c}")

    X, y = loader.load_all()
    print(f"\nRaw loaded: {X.shape[0]:,} flows × {X.shape[1]} features")

    # Remove classes with fewer than 5 samples (can't stratify)
    classes, counts = np.unique(y, return_counts=True)
    valid_classes   = classes[counts >= 5]
    mask            = np.isin(y, valid_classes)
    X, y            = X[mask], y[mask]

    # Stratified sample for QML simulator
    if n_samples < len(X):
        _, X, _, y = train_test_split(
            X, y,
            test_size=n_samples / len(X),
            stratify=y,
            random_state=42,
        )
        print(f"Sampled: {len(X):,} flows")

    return X, y, loader


# ═══════════════════════════════════════════════════════════
# STEP 2 — CLEAN & PREPROCESS
# ═══════════════════════════════════════════════════════════
def clean_data(X: np.ndarray, y: np.ndarray):
    print("\n" + "="*60)
    print("  STEP 2: Cleaning & Preprocessing")
    print("="*60)

    original_rows, original_cols = X.shape

    # 1. Replace Inf → NaN, drop rows with any NaN
    X = np.where(np.isinf(X), np.nan, X)
    nan_rows = np.any(np.isnan(X), axis=1)
    X, y = X[~nan_rows], y[~nan_rows]
    print(f"  Removed {nan_rows.sum():,} NaN/Inf rows  "
          f"({len(X):,} remain)")

    # 2. Remove zero-variance columns (constant features — useless for ML)
    variance    = np.var(X, axis=0)
    nonzero_var = variance > 1e-10
    X           = X[:, nonzero_var]
    print(f"  Removed {(~nonzero_var).sum()} zero-variance features  "
          f"({X.shape[1]} remain)")

    # 3. Clip outliers at mean ± 5σ per feature (winsorization)
    means = np.mean(X, axis=0)
    stds  = np.std(X,  axis=0) + 1e-9
    X     = np.clip(X, means - 5 * stds, means + 5 * stds)
    print(f"  Outlier capping: clipped values beyond mean ± 5σ")

    # 4. Class distribution report
    print(f"\n  Final: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"\n  Class distribution:")
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        pct = cnt / len(y) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:35s} {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    return X.astype(np.float32), y


# ═══════════════════════════════════════════════════════════
# STEP 3 — QUANTUM ENCODING DEMO
# ═══════════════════════════════════════════════════════════
def show_encoding(X: np.ndarray):
    print("\n" + "="*60)
    print("  STEP 3: Quantum Encoding Demo")
    print("="*60)
    try:
        import cirq
        enc = QuantumEncoder(backend="cirq", fit_scaler=True)
        enc.fit(X[:100])
        circuits = enc.encode_batch(X[:3], verbose=False)
        print(f"  Encoded 3 flows into Cirq circuits successfully.")
        print(f"  Each flow → 1 qubit with {X.shape[1]} sequential Rx gates.")

        # Show a compact 6-gate demo circuit
        q    = cirq.GridQubit(1, 1)
        demo = cirq.Circuit()
        vals = enc._transform(X[:1]).flatten()
        for v in vals[:6]:
            demo.append(cirq.rx(float(v) * np.pi)(q))
        print(f"\n  Sample circuit (6 of {X.shape[1]} Rx gates):")
        print(demo)
    except Exception as e:
        print(f"  Encoding demo skipped: {e}")


# ═══════════════════════════════════════════════════════════
# STEP 4 — HYPERPARAMETER OPTIMIZER
# ═══════════════════════════════════════════════════════════
def optimize_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fast: bool = True,
) -> dict:
    """
    Grid search over SVM and QSVM hyperparameters via 3-fold CV.

    Grid:
      SVM  : kernel ∈ {rbf, linear}, C ∈ {0.1, 1, 10}
      QSVM : n_qubits ∈ {2, 4},      C ∈ {0.1, 1, 10}

    In fast/demo mode the grid is smaller for speed.
    """
    print("\n" + "="*60)
    print("  STEP 4: Hyperparameter Optimization")
    print("="*60)

    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import GridSearchCV

    scaler   = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_train)
    le       = LabelEncoder()
    y_enc    = le.fit_transform(y_train)

    # ── SVM grid search ─────────────────────────────────────
    print("\n  [4a] Optimizing Classical SVM …")
    svm_grid = (
        {"C": [0.1, 1.0], "kernel": ["rbf"]}
        if fast else
        {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}
    )
    svm_gs = GridSearchCV(
        SVC(probability=True),
        svm_grid,
        cv=3, scoring="accuracy",
        n_jobs=-1, verbose=0,
    )
    svm_gs.fit(X_scaled, y_enc)
    best_svm = svm_gs.best_params_
    print(f"  Best SVM  → {best_svm}  |  CV acc: {svm_gs.best_score_*100:.2f}%")

    # Print full grid results
    print(f"\n  SVM Grid Results:")
    for params, mean_score in zip(
        svm_gs.cv_results_["params"],
        svm_gs.cv_results_["mean_test_score"],
    ):
        print(f"    {str(params):<45} acc={mean_score*100:.2f}%")

    # ── QSVM grid search ────────────────────────────────────
    print("\n  [4b] Optimizing QSVM (quantum kernel) …")
    qsvm_n_qubits = [4]        if fast else [2, 4]
    qsvm_C_vals   = [0.1, 1.0] if fast else [0.1, 1.0, 10.0]

    best_qsvm_acc    = -1
    best_qsvm_params = {"n_qubits": 4, "C": 1.0}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f"\n  QSVM Grid Results:")
    for n_q in qsvm_n_qubits:
        for C_val in qsvm_C_vals:
            fold_accs = []
            for tr_idx, val_idx in skf.split(X_train, y_train):
                try:
                    m = QSVMDetector(backend="cirq_sim", n_qubits=n_q, C=C_val)
                    m.fit(X_train[tr_idx], y_train[tr_idx])
                    preds = m.predict(X_train[val_idx])
                    ytrue = m.le.transform(y_train[val_idx])
                    fold_accs.append(accuracy_score(ytrue, preds))
                except Exception as ex:
                    fold_accs.append(0.0)

            mean_acc = np.mean(fold_accs)
            marker   = " ← best" if mean_acc > best_qsvm_acc else ""
            print(f"    n_qubits={n_q}  C={C_val:<5}  acc={mean_acc*100:.2f}%{marker}")

            if mean_acc > best_qsvm_acc:
                best_qsvm_acc    = mean_acc
                best_qsvm_params = {"n_qubits": n_q, "C": C_val}

    print(f"\n  Best QSVM → {best_qsvm_params}  |  CV acc: {best_qsvm_acc*100:.2f}%")

    return {"svm": best_svm, "qsvm": best_qsvm_params}


# ═══════════════════════════════════════════════════════════
# STEP 5 — TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════
def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    best_params: dict,
    ev: Evaluator,
    run_cnn: bool = False,
) -> dict:
    print("\n" + "="*60)
    print("  STEP 5: Training & Evaluation")
    print("="*60)
    results = {}

    # ── Classical SVM ────────────────────────────────────────
    print("\n  [5a] Classical SVM")
    svm = SVMBaseline(
        kernel=best_params["svm"].get("kernel", "rbf"),
        C=best_params["svm"].get("C", 1.0),
    )
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_true_svm = svm.le.transform(y_test)
    acc_svm    = accuracy_score(y_true_svm, y_pred_svm)
    print(f"\n  Accuracy: {acc_svm*100:.2f}%")
    print(svm.report(X_test, y_test))

    try:
        y_prob_svm = svm.model.predict_proba(svm._preprocess(X_test))
        ev.plot_combined(y_true_svm, y_pred_svm, y_prob_svm, title="SVM_IoT23")
    except Exception:
        ev.plot_confusion(y_true_svm, y_pred_svm, title="SVM_IoT23")
    results["SVM"] = {"model": svm, "acc": acc_svm, "time": svm.train_time}

    # ── QSVM ─────────────────────────────────────────────────
    print("\n  [5b] QSVM")
    qsvm = QSVMDetector(
        backend="cirq_sim",
        n_qubits=best_params["qsvm"].get("n_qubits", 4),
        C=best_params["qsvm"].get("C", 1.0),
    )
    qsvm.fit(X_train, y_train)
    y_pred_qsvm = qsvm.predict(X_test)
    y_true_qsvm = qsvm.le.transform(y_test)
    acc_qsvm    = accuracy_score(y_true_qsvm, y_pred_qsvm)
    print(f"\n  Accuracy: {acc_qsvm*100:.2f}%")
    print(qsvm.report(X_test, y_test))

    try:
        y_prob_qsvm = qsvm.predict_proba(X_test)
        ev.plot_combined(y_true_qsvm, y_pred_qsvm, y_prob_qsvm, title="QSVM_IoT23")
    except Exception:
        ev.plot_confusion(y_true_qsvm, y_pred_qsvm, title="QSVM_IoT23")
    results["QSVM"] = {"model": qsvm, "acc": acc_qsvm, "time": qsvm.train_time}

    # ── QCNN (optional) ──────────────────────────────────────
    if run_cnn:
        print("\n  [5c] QCNN")
        try:
            qcnn = QCNNDetector(
                n_qubits=4, n_layers=1,
                backend="cirq_sim",
                epochs=10, batch_size=16,
            )
            qcnn.fit(X_train, y_train)
            y_pred_qcnn = qcnn.predict(X_test)
            y_true_qcnn = qcnn.le.transform(y_test)
            acc_qcnn    = accuracy_score(y_true_qcnn, y_pred_qcnn)
            print(f"\n  Accuracy: {acc_qcnn*100:.2f}%")
            print(qcnn.report(X_test, y_test))
            ev.plot_confusion(y_true_qcnn, y_pred_qcnn, title="QCNN_IoT23")
            results["QCNN"] = {"model": qcnn, "acc": acc_qcnn,
                                "time": qcnn.train_time}
        except Exception as e:
            print(f"  QCNN failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════
# STEP 6 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
def print_summary(results: dict):
    print("\n" + "="*60)
    print("  STEP 6: Final Summary")
    print("="*60)
    print(f"\n  {'Model':<10} {'Accuracy':>10}  {'Train Time':>12}")
    print(f"  {'─'*36}")
    for name, info in results.items():
        t = info.get("time") or 0
        print(f"  {name:<10} {info['acc']*100:>9.2f}%  {t:>10.2f}s")

    best = max(results, key=lambda k: results[k]["acc"])
    print(f"\n  ✓ Best model: {best}  ({results[best]['acc']*100:.2f}% accuracy)")
    print(f"\n  Plots saved in: results/")


# ═══════════════════════════════════════════════════════════
# STEP 7 — SAVE MODELS
# ═══════════════════════════════════════════════════════════
def save_models(results: dict, X_test, y_test, save_dir: str = "results"):
    print("\n" + "="*60)
    print("  STEP 7: Saving Models")
    print("="*60)

    for name, info in results.items():
        model_path = os.path.join(save_dir, f"{name.lower()}_model.joblib")
        joblib.dump(info["model"], model_path)
        print(f"  ✓ {name} model saved → {model_path}")

    # Save test data for later verification
    test_data_path = os.path.join(save_dir, "test_data.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)
    print(f"  ✓ Test data saved  → {test_data_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="QML Intrusion Detection — IoT-23 Full Pipeline"
    )
    parser.add_argument("--iot23", default=None,
        help="Path to folder containing CTU-IoT-Malware-Capture-* subfolders")
    parser.add_argument("--demo", action="store_true",
        help="Run on synthetic data (no dataset needed)")
    parser.add_argument("--mode", default="demo",
        choices=["demo", "full"],
        help="demo=SVM+QSVM only  |  full=+QCNN+benchmark")
    parser.add_argument("--n-samples", type=int, default=1000,
        help="Flows to use for QML training (default 1000)")
    parser.add_argument("--binary", action="store_true",
        help="Binary labels: Benign vs Malicious")
    parser.add_argument("--skip-optimize", action="store_true",
        help="Skip optimizer, use default hyperparams")
    parser.add_argument("--results", default="results",
        help="Directory to save evaluation plots")
    args = parser.parse_args()

    print("=" * 60)
    print("  QML-Based Intrusion Detection  |  IoT-23 Dataset")
    print("  Paper: Kalinin & Krundyshev, 2022")
    print("=" * 60)

    os.makedirs(args.results, exist_ok=True)

    # ── Load ────────────────────────────────────────────────
    if args.iot23:
        X, y, _     = load_iot23(args.iot23, args.n_samples, args.binary)
        class_names = sorted(set(y))
    else:
        print("\n[Pipeline] --demo mode: using synthetic data.")
        df          = StreamBuilder.generate_synthetic(n_streams=args.n_samples)
        feat_cols   = [c for c in df.columns if c != "label"]
        X           = df[feat_cols].values.astype(np.float32)
        y           = df["label"].values
        class_names = sorted(set(y))

    # ── Clean ───────────────────────────────────────────────
    X, y = clean_data(X, y)

    # ── Encode demo ─────────────────────────────────────────
    show_encoding(X)

    # ── Split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    ev = Evaluator(class_names=class_names, save_dir=args.results)

    # ── Optimize ────────────────────────────────────────────
    if args.skip_optimize:
        best_params = {
            "svm":  {"kernel": "rbf", "C": 1.0},
            "qsvm": {"n_qubits": 4,   "C": 1.0},
        }
        print(f"\n  [Optimizer] Skipped → defaults: {best_params}")
    else:
        best_params = optimize_hyperparams(
            X_train, y_train, fast=(args.mode == "demo")
        )

    # ── Train & Evaluate ────────────────────────────────────
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        best_params, ev,
        run_cnn=(args.mode == "full"),
    )

    # ── Time benchmark (full mode) ──────────────────────────
    if args.mode == "full":
        print("\n" + "="*60)
        print("  BENCHMARK: Training Time vs Dataset Size  (Table 4)")
        print("="*60)
        models  = {"SVM": SVMBaseline(), "QSVM": QSVMDetector(backend="cirq_sim", n_qubits=4)}
        sizes   = [50, 100, 200, 500, 1000]
        bench   = ev.benchmark_training_time(models, X, y, sizes=sizes)
        ev.print_table4(bench, sizes)

    # ── Summary ─────────────────────────────────────────────
    print_summary(results)

    # ── Save models ─────────────────────────────────────────
    save_models(results, X_test, y_test, save_dir=args.results)


if __name__ == "__main__":
    main()
