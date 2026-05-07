"""
IBM Quantum QSVM — Full IoT-23 Pipeline
=========================================
Runs QSVM on real IBM Quantum hardware using Qiskit Runtime.

Architecture:
  - ZZFeatureMap for quantum feature encoding (entangled kernel)
  - FidelityQuantumKernel via Sampler primitive
  - SVC with precomputed quantum kernel matrix
  - Nyström approximation for large datasets (>2000 samples)

Run:
  python run_ibm_qsvm.py \
    --iot23 /path/to/IoTScenarios \
    --token  YOUR_IBM_TOKEN \
    --backend ibm_brisbane \
    --n-samples 5000 \
    --n-qubits 4

Options:
  --token      STR   IBM Quantum API token (required)
  --backend    STR   IBM backend name      (default: least_busy)
  --n-samples  INT   training samples      (default: 5000)
  --n-qubits   INT   qubits / PCA dims     (default: 4)
  --reps       INT   ZZFeatureMap reps     (default: 2)
  --C          FLOAT SVC regularisation    (default: 1.0)
  --landmarks  INT   Nyström landmarks     (default: 800)
  --simulator        use IBM Aer simulator (no token needed)
  --results    DIR   output directory      (default: results_ibm)
"""

import sys, os, warnings, time, joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.iot23_loader import IoT23Loader
from utils.evaluation import Evaluator

import argparse

parser = argparse.ArgumentParser(description="IBM Quantum QSVM — IoT-23")
parser.add_argument("--iot23",      required=True,
                    help="Path to IoTScenarios folder")
parser.add_argument("--token",      default=None,
                    help="IBM Quantum API token")
parser.add_argument("--backend",    default=None,
                    help="IBM backend name (e.g. ibm_brisbane). "
                         "Default: least busy available backend")
parser.add_argument("--n-samples",  type=int,   default=5000,
                    help="Total samples to use (default 5000)")
parser.add_argument("--n-qubits",   type=int,   default=4,
                    help="Qubits = PCA dimensions (default 4)")
parser.add_argument("--reps",       type=int,   default=2,
                    help="ZZFeatureMap repetitions (default 2)")
parser.add_argument("--C",          type=float, default=1.0,
                    help="SVC C parameter (default 1.0)")
parser.add_argument("--landmarks",  type=int,   default=800,
                    help="Nyström landmark points (default 800)")
parser.add_argument("--simulator",  action="store_true",
                    help="Use IBM Aer simulator (no token needed)")
parser.add_argument("--binary",     action="store_true",
                    help="Binary: Benign vs Malicious")
parser.add_argument("--results",    default="results_ibm")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)

print("=" * 60)
print("  IBM Quantum QSVM — IoT-23 Full Dataset")
print("=" * 60)
print(f"  backend    = {args.backend or 'least_busy'}")
print(f"  n_samples  = {args.n_samples:,}")
print(f"  n_qubits   = {args.n_qubits}")
print(f"  reps       = {args.reps}")
print(f"  C          = {args.C}")
print(f"  landmarks  = {args.landmarks}")
print(f"  simulator  = {args.simulator}")
print(f"  binary     = {args.binary}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CONNECT TO IBM QUANTUM
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1: Connecting to IBM Quantum")
print("="*60)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from qiskit_ibm_runtime import SamplerV2 as SamplerV2

    if args.simulator:
        # Use local Aer simulator — no token needed
        from qiskit_aer.primitives import Sampler as AerSampler
        sampler  = AerSampler()
        service  = None
        backend  = None
        backend_name = "aer_simulator"
        print("  Using IBM Aer simulator (local, no token needed)")

    else:
        if not args.token:
            print("\n  ERROR: --token is required for real IBM hardware.")
            print("  Get your token from: https://quantum.ibm.com/account")
            print("  Or use --simulator flag for local simulation.")
            sys.exit(1)

        # Save token (only needed first time)
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum",
                token=args.token,
                overwrite=True,
            )
            print("  Token saved successfully.")
        except Exception as e:
            print(f"  Token save warning: {e}")

        service = QiskitRuntimeService(channel="ibm_quantum")

        # Select backend
        if args.backend:
            backend = service.backend(args.backend)
        else:
            # Automatically pick least busy backend
            backends     = service.backends(
                filters=lambda b: (
                    b.status().operational and
                    b.num_qubits >= args.n_qubits and
                    not b.configuration().simulator
                )
            )
            backend      = min(backends,
                               key=lambda b: b.status().pending_jobs)
            args.backend = backend.name

        backend_name = backend.name
        print(f"  Connected to: {backend_name}")
        print(f"  Qubits available: {backend.num_qubits}")
        print(f"  Pending jobs: {backend.status().pending_jobs}")

except ImportError as e:
    print(f"\n  ImportError: {e}")
    print("\n  Install Qiskit IBM Runtime:")
    print("  pip install qiskit-ibm-runtime qiskit-machine-learning qiskit-aer")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# STEP 2 — LOAD FULL IoT-23 DATASET
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2: Loading Full IoT-23 Dataset")
print("="*60)

loader = IoT23Loader(
    dataset_dir=args.iot23,
    binary=args.binary,
    max_rows_per_capture=None,    # Load ALL rows
)
X_raw, y_raw = loader.load_all()
print(f"\n  Loaded: {X_raw.shape[0]:,} flows × {X_raw.shape[1]} features")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — CLEAN
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3: Cleaning & Preprocessing")
print("="*60)

# NaN / Inf
X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
mask  = ~np.any(np.isnan(X_raw), axis=1)
X_raw, y_raw = X_raw[mask], y_raw[mask]

# Zero-variance
var_mask = np.var(X_raw, axis=0) > 1e-10
X_raw    = X_raw[:, var_mask]

# Outlier clipping
means = np.mean(X_raw, axis=0)
stds  = np.std(X_raw,  axis=0) + 1e-9
X_raw = np.clip(X_raw, means - 5*stds, means + 5*stds).astype(np.float32)

# Remove tiny classes
classes, counts = np.unique(y_raw, return_counts=True)
valid_cls       = classes[counts >= 50]
mask            = np.isin(y_raw, valid_cls)
X_raw, y_raw    = X_raw[mask], y_raw[mask]

print(f"  Final: {X_raw.shape[0]:,} flows × {X_raw.shape[1]} features")
print(f"\n  Class distribution:")
for c, n in zip(*np.unique(y_raw, return_counts=True)):
    pct = n / len(y_raw) * 100
    bar = "█" * int(pct / 2)
    print(f"    {c:35s} {n:>8,}  ({pct:5.1f}%)  {bar}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — SPLIT & PREPROCESS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4: Splitting & Feature Preprocessing")
print("="*60)

# Full train/test split on ALL data
X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(
    X_raw, y_raw,
    test_size=0.2,
    stratify=y_raw,
    random_state=42,
)
print(f"  Full train: {len(X_tr_full):,}  |  Full test: {len(X_te_full):,}")

# Scale to [0, π]
scaler    = MinMaxScaler(feature_range=(0, np.pi))
X_tr_s    = scaler.fit_transform(X_tr_full)
X_te_s    = scaler.transform(X_te_full)

# PCA → n_qubits dimensions
pca       = PCA(n_components=args.n_qubits)
X_tr_pca  = pca.fit_transform(X_tr_s)
X_te_pca  = pca.transform(X_te_s)

# Clip to [0, π] after PCA
X_tr_pca  = np.clip(X_tr_pca, 0, np.pi)
X_te_pca  = np.clip(X_te_pca, 0, np.pi)

le        = LabelEncoder()
y_tr_enc  = le.fit_transform(y_tr_full)
y_te_enc  = le.transform(y_te_full)

print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  Features after PCA: {X_tr_pca.shape[1]}")
print(f"  Classes: {list(le.classes_)}")

# Stratified sample for QSVM kernel computation
# (Nyström uses landmarks, train SVC on full projected features)
if args.n_samples < len(X_tr_pca):
    _, X_qsvm_tr, _, y_qsvm_tr = train_test_split(
        X_tr_pca, y_tr_enc,
        test_size=args.n_samples / len(X_tr_pca),
        stratify=y_tr_enc,
        random_state=42,
    )
else:
    X_qsvm_tr, y_qsvm_tr = X_tr_pca, y_tr_enc

print(f"\n  QSVM training on: {len(X_qsvm_tr):,} samples")
print(f"  Nyström landmarks: {args.landmarks}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — BUILD QUANTUM KERNEL
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5: Building IBM Quantum Kernel")
print("="*60)

from qiskit.circuit.library import ZZFeatureMap

feature_map = ZZFeatureMap(
    feature_dimension=args.n_qubits,
    reps=args.reps,
    entanglement="linear",    # linear entanglement between adjacent qubits
)
print(f"\n  ZZFeatureMap circuit:")
print(f"    feature_dimension = {args.n_qubits}")
print(f"    reps              = {args.reps}")
print(f"    entanglement      = linear")
print(f"    circuit depth     = {feature_map.decompose().depth()}")
print(f"\n  Circuit diagram (first rep):")
print(feature_map.decompose().draw(output='text', fold=80))

# Build quantum kernel
if args.simulator:
    from qiskit_aer.primitives import Sampler as AerSampler
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    fidelity = ComputeUncompute(sampler=AerSampler())
    q_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map,
    )
    print("\n  Quantum kernel: FidelityQuantumKernel (Aer simulator)")

else:
    from qiskit_ibm_runtime import Session, SamplerV2
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    session  = Session(service=service, backend=backend)
    sampler  = SamplerV2(session=session)
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(
        fidelity=fidelity,
        feature_map=feature_map,
    )
    print(f"\n  Quantum kernel: FidelityQuantumKernel (IBM {backend_name})")
    print(f"  Session opened on {backend_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — CLASSICAL SVM BASELINE (full data)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6: Classical SVM Baseline (Full Dataset)")
print("="*60)

print(f"\n  Training SVM on {len(X_tr_pca):,} samples ...")
t0    = time.time()
svm   = SVC(kernel="rbf", C=args.C, probability=True, class_weight="balanced")
svm.fit(X_tr_pca, y_tr_enc)
svm_train_time = time.time() - t0

y_pred_svm = svm.predict(X_te_pca)
acc_svm    = accuracy_score(y_te_enc, y_pred_svm)

print(f"\n  SVM Accuracy : {acc_svm*100:.2f}%")
print(f"  SVM Train Time: {svm_train_time:.2f}s")
print(classification_report(y_te_enc, y_pred_svm,
      target_names=le.classes_, zero_division=0))

joblib.dump({
    "model": svm, "scaler": scaler, "pca": pca, "le": le,
    "accuracy": acc_svm, "train_time": svm_train_time,
}, f"{args.results}/svm_full.joblib")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — QSVM WITH NYSTRÖM APPROXIMATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 7: IBM Quantum QSVM (Nyström Approximation)")
print("="*60)

# ── Select landmark points ────────────────────────────────────
n_landmarks  = min(args.landmarks, len(X_qsvm_tr))
landmark_idx = np.random.choice(len(X_qsvm_tr),
                                 n_landmarks,
                                 replace=False)
X_landmarks  = X_qsvm_tr[landmark_idx]

print(f"\n  Computing quantum kernel matrix...")
print(f"  Training samples : {len(X_qsvm_tr):,}")
print(f"  Landmark points  : {n_landmarks}")
print(f"  Kernel evaluations needed:")
print(f"    Train×Landmarks: {len(X_qsvm_tr):,} × {n_landmarks} "
      f"= {len(X_qsvm_tr)*n_landmarks:,}")
print(f"    Landmarks×Landmarks: {n_landmarks} × {n_landmarks} "
      f"= {n_landmarks**2:,}")

# ── Compute landmark-landmark kernel ─────────────────────────
print(f"\n  [1/2] Computing K(landmarks, landmarks) ...")
t0           = time.time()
K_land_land  = q_kernel.evaluate(x_vec=X_landmarks)
t_land       = time.time() - t0
print(f"  Done in {t_land:.1f}s  "
      f"(shape: {K_land_land.shape})")

# ── Compute train-landmark kernel ────────────────────────────
print(f"\n  [2/2] Computing K(train, landmarks) ...")
# Process in batches to avoid timeout on IBM hardware
BATCH_SIZE    = 200
K_train_parts = []
n_batches     = int(np.ceil(len(X_qsvm_tr) / BATCH_SIZE))

t0 = time.time()
for b in range(n_batches):
    start = b * BATCH_SIZE
    end   = min(start + BATCH_SIZE, len(X_qsvm_tr))
    batch = X_qsvm_tr[start:end]

    K_batch = q_kernel.evaluate(x_vec=batch, y_vec=X_landmarks)
    K_train_parts.append(K_batch)

    elapsed = time.time() - t0
    pct     = (b + 1) / n_batches * 100
    bar     = "█" * int(pct / 5)
    eta     = elapsed / (b + 1) * (n_batches - b - 1)
    print(f"  Batch {b+1:>3}/{n_batches}  [{bar:<20}] "
          f"{pct:5.1f}%  ETA: {eta:.0f}s", end="\r")

K_train_land  = np.vstack(K_train_parts)
t_kernel      = time.time() - t0
print(f"\n  Kernel computation done in {t_kernel:.1f}s")

# ── Nyström approximation ─────────────────────────────────────
print("\n  Computing Nyström approximation ...")
from numpy.linalg import pinv

# Full kernel approximation: K ≈ K_train_land @ K_land_land^+ @ K_train_land.T
K_land_inv   = pinv(K_land_land)
K_train_full = K_train_land @ K_land_inv @ K_train_land.T
print(f"  Approximated kernel matrix: {K_train_full.shape}")

# ── Train SVM on quantum kernel ───────────────────────────────
print("\n  Training SVM on quantum kernel matrix ...")
t0     = time.time()
qsvm   = SVC(kernel="precomputed", C=args.C,
             probability=True, class_weight="balanced")
qsvm.fit(K_train_full, y_qsvm_tr)
qsvm_train_time = time.time() - t0
print(f"  SVM fit done in {qsvm_train_time:.2f}s")

# ── Predict on test set ───────────────────────────────────────
print("\n  Computing test kernel (batched) ...")
K_test_parts = []
n_te_batches = int(np.ceil(len(X_te_pca) / BATCH_SIZE))

t0 = time.time()
for b in range(n_te_batches):
    start   = b * BATCH_SIZE
    end     = min(start + BATCH_SIZE, len(X_te_pca))
    K_batch = q_kernel.evaluate(x_vec=X_te_pca[start:end],
                                 y_vec=X_landmarks)
    K_test_parts.append(K_batch)
    print(f"  Test batch {b+1}/{n_te_batches}", end="\r")

K_test_land  = np.vstack(K_test_parts)
K_test_full  = K_test_land @ K_land_inv @ \
               q_kernel.evaluate(x_vec=X_landmarks,
                                  y_vec=X_qsvm_tr).T

y_pred_qsvm  = qsvm.predict(K_test_full)
acc_qsvm     = accuracy_score(y_te_enc, y_pred_qsvm)
total_time   = t_kernel + qsvm_train_time

print(f"\n\n  QSVM Accuracy : {acc_qsvm*100:.2f}%")
print(f"  Kernel Time   : {t_kernel:.1f}s")
print(f"  SVM Fit Time  : {qsvm_train_time:.2f}s")
print(f"  Total Time    : {total_time:.1f}s")
print(classification_report(y_te_enc, y_pred_qsvm,
      target_names=le.classes_, zero_division=0))


# ═══════════════════════════════════════════════════════════════
# STEP 8 — EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 8: Evaluation Plots")
print("="*60)

ev = Evaluator(class_names=list(le.classes_), save_dir=args.results)

# SVM plots
try:
    y_prob_svm = svm.predict_proba(X_te_pca)
    ev.plot_combined(y_te_enc, y_pred_svm, y_prob_svm,
                     title=f"SVM_Full_IoT23")
except Exception as e:
    ev.plot_confusion(y_te_enc, y_pred_svm, title="SVM_Full_IoT23")

# QSVM plots
try:
    y_prob_qsvm = qsvm.predict_proba(K_test_full)
    ev.plot_combined(y_te_enc, y_pred_qsvm, y_prob_qsvm,
                     title=f"QSVM_IBM_{backend_name}")
except Exception as e:
    ev.plot_confusion(y_te_enc, y_pred_qsvm,
                      title=f"QSVM_IBM_{backend_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 9 — SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════
joblib.dump({
    "qsvm":          qsvm,
    "scaler":        scaler,
    "pca":           pca,
    "le":            le,
    "feature_map":   feature_map,
    "X_landmarks":   X_landmarks,
    "K_land_land":   K_land_land,
    "K_land_inv":    K_land_inv,
    "backend":       backend_name,
    "n_qubits":      args.n_qubits,
    "accuracy":      acc_qsvm,
    "kernel_time":   t_kernel,
    "train_time":    qsvm_train_time,
}, f"{args.results}/qsvm_ibm_{backend_name}.joblib")

print(f"\n  Models saved to: {args.results}/")

# Close IBM session
if not args.simulator and service is not None:
    try:
        session.close()
        print("  IBM Quantum session closed.")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# STEP 10 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"\n  Backend: {backend_name}")
print(f"  Dataset: IoT-23 ({X_raw.shape[0]:,} total flows)")
print(f"  Classes: {list(le.classes_)}")
print(f"\n  {'Model':<25} {'Accuracy':>10}  {'Time':>10}")
print(f"  {'─'*50}")
print(f"  {'SVM (Full Dataset)':<25} {acc_svm*100:>9.2f}%  "
      f"{svm_train_time:>8.1f}s")
print(f"  {'QSVM (IBM Quantum)':<25} {acc_qsvm*100:>9.2f}%  "
      f"{total_time:>8.1f}s")

diff = acc_qsvm - acc_svm
sign = "+" if diff >= 0 else ""
print(f"\n  QSVM vs SVM: {sign}{diff*100:.2f}%")
print(f"\n  Plots saved:")
print(f"    {args.results}/combined_SVM_Full_IoT23.png")
print(f"    {args.results}/combined_QSVM_IBM_{backend_name}.png")
print(f"    {args.results}/qsvm_ibm_{backend_name}.joblib")