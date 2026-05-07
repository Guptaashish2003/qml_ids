"""
QCNN Training Script — IoT-23 Dataset (Full Research Suite)
==========================================================
Runs the full QCNN pipeline with research-optimized architecture:
  1. Load IoT-23 data (Step 1)
  2. Clean & preprocess (Step 2)
  3. Stratified Sampling & PCA Scaling (Step 3)
  4. Classical CNN Baseline Comparison (Step 4)
  5. Advanced QCNN Training (Step 5)
  6. Multi-Metric Evaluation (Step 6)
  7. Loss Curve Analysis (Step 7)
  8. Final Comparative Summary (Step 8)
"""

import sys, os, warnings, time, joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.iot23_loader import IoT23Loader
from utils.evaluation import Evaluator

import argparse

parser = argparse.ArgumentParser(description="QCNN Intrusion Detection — IoT-23")
parser.add_argument("--iot23",       required=True)
parser.add_argument("--n-samples",   type=int,   default=500)
parser.add_argument("--n-qubits",    type=int,   default=4)
parser.add_argument("--n-layers",    type=int,   default=1)
parser.add_argument("--epochs",      type=int,   default=30)
parser.add_argument("--batch-size",  type=int,   default=32)
parser.add_argument("--lr",          type=float, default=0.05)
parser.add_argument("--binary",      action="store_true")
parser.add_argument("--results",     default="results_qcnn")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)

print("=" * 60)
print("  QCNN Intrusion Detection — IoT-23 (Full Research Suite)")
print("=" * 60)
print(f"  n_qubits  = {args.n_qubits}")
print(f"  n_layers  = {args.n_layers}")
print(f"  epochs    = {args.epochs}")
print(f"  batch     = {args.batch_size}")
print(f"  lr        = {args.lr}")
print(f"  n_samples = {args.n_samples}")
print(f"  binary    = {args.binary}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1: Loading IoT-23 Dataset")
print("="*60)

loader = IoT23Loader(
    dataset_dir=args.iot23,
    binary=args.binary,
    max_rows_per_capture=max(500, (args.n_samples // 20) * 5),
)
X_raw, y_raw = loader.load_all()
print(f"\nLoaded: {X_raw.shape[0]:,} flows × {X_raw.shape[1]} features")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — CLEANING & DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2: Cleaning")
print("="*60)

X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
mask  = ~np.any(np.isnan(X_raw), axis=1)
X_raw, y_raw = X_raw[mask], y_raw[mask]

var_mask = np.var(X_raw, axis=0) > 1e-10
X_raw    = X_raw[:, var_mask]

means = np.mean(X_raw, axis=0)
stds  = np.std(X_raw,  axis=0) + 1e-9
X_raw = np.clip(X_raw, means - 5*stds, means + 5*stds).astype(np.float32)

classes, counts = np.unique(y_raw, return_counts=True)
valid_cls = classes[counts >= 5]
mask      = np.isin(y_raw, valid_cls)
X_raw, y_raw = X_raw[mask], y_raw[mask]

print(f"  Shape after cleaning: {X_raw.shape}")
print(f"\n  Class distribution:")
for c, n in zip(*np.unique(y_raw, return_counts=True)):
    bar = "█" * int(n / len(y_raw) * 40)
    print(f"    {c:35s} {n:>7,}  ({n/len(y_raw)*100:5.1f}%)  {bar}")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — SAMPLING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3: Sampling & Splitting")
print("="*60)

if args.n_samples < len(X_raw):
    _, X, _, y = train_test_split(
        X_raw, y_raw, test_size=args.n_samples / len(X_raw),
        stratify=y_raw, random_state=42
    )
else:
    X, y = X_raw, y_raw

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize first (better for PCA)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)

le = LabelEncoder()
y_train_e = le.fit_transform(y_train)
y_test_e  = le.transform(y_test)
n_classes = len(le.classes_)

# PCA for dimensionality reduction
pca = PCA(n_components=args.n_qubits)
X_train_pca_raw = pca.fit_transform(X_train_std)
X_test_pca_raw  = pca.transform(X_test_std)

# Finally scale to [0, pi] for angle encoding
pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_pca = pca_scaler.fit_transform(X_train_pca_raw)
X_test_pca  = pca_scaler.transform(X_test_pca_raw)

print(f"  After PCA: {X_train_pca.shape[1]} features → {args.n_qubits} qubits")
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Keep the original scaled version for CNN
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

ev = Evaluator(class_names=list(le.classes_), save_dir=args.results)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — CLASSICAL CNN BASELINE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4: Classical CNN Baseline")
print("="*60)

acc_cnn = None
cnn_time = None

try:
    import tensorflow as tf
    def build_cnn(input_dim, n_classes):
        inp = tf.keras.Input(shape=(input_dim, 1))
        x   = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
        x   = tf.keras.layers.MaxPooling1D(2)(x)
        x   = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x   = tf.keras.layers.GlobalAveragePooling1D()(x)
        x   = tf.keras.layers.Dense(64, activation="relu")(x)
        x   = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        mdl = tf.keras.Model(inp, out)
        mdl.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return mdl

    X_cnn_train = X_train_s[:, :, np.newaxis]
    X_cnn_test  = X_test_s[:,  :, np.newaxis]
    cnn_model = build_cnn(X_train_s.shape[1], n_classes)
    
    t0 = time.time()
    cnn_model.fit(X_cnn_train, y_train_e, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    cnn_time = time.time() - t0
    
    y_prob_cnn = cnn_model.predict(X_cnn_test, verbose=0)
    y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
    acc_cnn = accuracy_score(y_test_e, y_pred_cnn)
    print(f"  CNN Accuracy : {acc_cnn*100:.2f}%")
    print(classification_report(y_test_e, y_pred_cnn, target_names=le.classes_, zero_division=0))
    ev.plot_combined(y_test_e, y_pred_cnn, y_prob_cnn, title="CNN_IoT23")

except ImportError:
    print("  TensorFlow not installed — skipping CNN baseline.")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — QCNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5: QCNN Training (Research Optimized)")
print("="*60)

import cirq

def encode_input(x, n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        val = float(x[i]) if i < len(x) else 0.0
        circuit.append(cirq.rx(val)(q))
        circuit.append(cirq.ry(val)(q))
    return circuit, qubits

def conv_layer(qubits, params, offset=0):
    """Strongly Entangling Layer + ZZ Interaction (11 params per pair)."""
    ops = []
    p   = offset
    for i in range(0, len(qubits) - 1, 2):
        q0, q1 = qubits[i], qubits[i+1]
        ops += [
            # ZZ Interaction (Phase encoding of correlations)
            cirq.ZZ(q0, q1) ** params[p],
            # Local rotations
            cirq.ry(params[p+1])(q0), cirq.rz(params[p+2])(q0), cirq.ry(params[p+3])(q0),
            cirq.ry(params[p+4])(q1), cirq.rz(params[p+5])(q1), cirq.ry(params[p+6])(q1),
            cirq.CNOT(q0, q1),
            cirq.ry(params[p+7])(q1), cirq.rz(params[p+8])(q1), cirq.ry(params[p+9])(q1),
            cirq.rz(params[p+10])(q1)
        ]
        p += 11
    return ops, p - offset

def pool_layer(qubits, params, offset=0):
    ops, sources, sinks = [], qubits[::2], qubits[1::2]
    p = offset
    for src, snk in zip(sources, sinks):
        ops += [cirq.CNOT(src, snk), cirq.rz(params[p])(snk)]
        p += 1
    return ops, sinks, p - offset

def build_qcnn_circuit(x, params, n_qubits, n_layers):
    """Advanced Circuit with Data Re-uploading."""
    qubits = cirq.LineQubit.range(n_qubits)
    main_circuit = cirq.Circuit()
    active = list(qubits)
    param_idx = 0
    
    # Init Encoding
    enc0, _ = encode_input(x, n_qubits)
    main_circuit.append(enc0)
    for q in active: main_circuit.append(cirq.H(q))

    for layer in range(n_layers):
        ops, consumed = conv_layer(active, params, offset=param_idx)
        main_circuit.append(ops)
        param_idx += consumed
        ops, active, consumed = pool_layer(active, params, offset=param_idx)
        main_circuit.append(ops)
        param_idx += consumed
        # Data Re-uploading
        if len(active) > 1:
            for i, q in enumerate(active):
                val = float(x[i % len(x)])
                main_circuit.append(cirq.rx(val)(q))
                main_circuit.append(cirq.ry(val)(q))

    for q in active:
        main_circuit.append(cirq.ry(params[param_idx])(q))
        main_circuit.append(cirq.rz(params[param_idx+1])(q))
        param_idx += 2

    return main_circuit, active[-1], param_idx

def count_params(n_qubits, n_layers):
    total, active = 0, n_qubits
    for _ in range(n_layers):
        n_pairs = active // 2
        total += n_pairs * 11 + n_pairs  # conv + pool
        active //= 2
    total += active * 2
    return total

_sim = cirq.Simulator()

def forward(x, params, n_qubits, n_layers):
    circuit, readout_q, _ = build_qcnn_circuit(x, params, n_qubits, n_layers)
    res = _sim.simulate(circuit)
    probs = np.abs(res.final_state_vector) ** 2
    idx = readout_q.x
    exp_z = sum(probs[s] * (1 - 2 * ((s >> (n_qubits - 1 - idx)) & 1)) for s in range(2**n_qubits))
    return float(exp_z)

def compute_batch_grads(X_batch, y_batch, params, n_qubits, n_layers):
    n_p, grads, total_loss = len(params), np.zeros(len(params)), 0.0
    for x, y_tgt in zip(X_batch, y_batch):
        pred = forward(x, params, n_qubits, n_layers)
        total_loss += 0.5 * (pred - y_tgt) ** 2
        err = pred - y_tgt
        for k in range(n_p):
            orig = params[k]
            params[k] = orig + np.pi/2
            fp = forward(x, params, n_qubits, n_layers)
            params[k] = orig - np.pi/2
            fm = forward(x, params, n_qubits, n_layers)
            params[k] = orig
            grads[k] += (fp - fm) / 2.0 * err
    return grads / len(X_batch), total_loss / len(X_batch)

# --- Adam + LR Logic ---
def adam_update(params, grads, m, v, t, lr):
    t += 1
    m = 0.9 * m + 0.1 * grads
    v = 0.999 * v + 0.001 * grads**2
    m_h, v_h = m/(1-0.9**t), v/(1-0.999**t)
    params -= lr * m_h / (np.sqrt(v_h) + 1e-8)
    return m, v, t

def cosine_lr(base, epoch, total):
    return 1e-4 + 0.5 * (base - 1e-4) * (1 + np.cos(np.pi * epoch / total))

n_params = count_params(args.n_qubits, args.n_layers)
print(f"  QCNN Params: {n_params} | Batch: {args.batch_size} | Epochs: {args.epochs}")

rng = np.random.default_rng(42)
ovr_params, ovr_losses = [], []
t_qcnn_start = time.time()

for cls_idx in range(n_classes):
    cls_name = le.classes_[cls_idx]
    print(f"\n  ── Training Classifier: {cls_name} ──")
    y_bin = np.where(y_train_e == cls_idx, 1.0, -1.0)
    pos_idx, neg_idx = np.where(y_train_e == cls_idx)[0], np.where(y_train_e != cls_idx)[0]
    
    params = np.random.uniform(-0.1, 0.1, n_params)
    m, v, t_adam, losses = np.zeros(n_params), np.zeros(n_params), 0, []
    best_loss, best_params, patience, no_improve = np.inf, params.copy(), 7, 0

    for epoch in range(args.epochs):
        lr_t = cosine_lr(args.lr, epoch, args.epochs)
        h = max(args.batch_size // 2, 1)
        b_idx = np.concatenate([rng.choice(pos_idx, size=min(h, len(pos_idx)), replace=True),
                               rng.choice(neg_idx, size=min(h, len(neg_idx)), replace=True)])
        grads, avg_loss = compute_batch_grads(X_train_pca[b_idx], y_bin[b_idx], params, args.n_qubits, args.n_layers)
        norm = np.linalg.norm(grads)
        if norm > 1.0: grads /= norm
        m, v, t_adam = adam_update(params, grads, m, v, t_adam, lr_t)
        losses.append(avg_loss)
        if avg_loss < best_loss - 1e-4:
            best_loss, best_params, no_improve = avg_loss, params.copy(), 0
        else: no_improve += 1
        bar = "█" * int((1 - min(avg_loss, 1)) * 20)
        print(f"    Epoch {epoch+1:>2}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_t:.4f} | {bar}")
        if no_improve >= patience and epoch > args.epochs // 2: break
    ovr_params.append(best_params)
    ovr_losses.append(losses)

qcnn_time = time.time() - t_qcnn_start


# ═══════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  STEP 6: Evaluation & Plots\n" + "="*60)

scores = np.zeros((len(X_test_pca), n_classes))
for cls_idx in range(n_classes):
    p = ovr_params[cls_idx]
    scores[:, cls_idx] = [forward(x, p, args.n_qubits, args.n_layers) for x in X_test_pca]

if n_classes == 2:
    # Research Refinement: Dynamic Threshold Tuning
    print("  Tuning decision threshold on training data ...")
    train_scores = np.zeros((len(X_train_pca), 2))
    for i in range(2):
        p = ovr_params[i]
        # Sample 200 train points for speed
        idx = rng.choice(len(X_train_pca), size=min(200, len(X_train_pca)), replace=False)
        train_scores[idx, i] = [forward(x, p, args.n_qubits, args.n_layers) for x in X_train_pca[idx]]
    
    # Differential score on train subset
    train_diff = train_scores[idx, 1] - train_scores[idx, 0]
    y_train_sub = y_train_e[idx]
    
    # Search for best threshold
    best_thr = 0.0
    best_acc = 0.0
    for thr in np.linspace(-1.0, 1.0, 41):
        tmp_acc = accuracy_score(y_train_sub, (train_diff > thr).astype(int))
        if tmp_acc > best_acc:
            best_acc = tmp_acc
            best_thr = thr
    
    print(f"  Optimal threshold found: {best_thr:.3f} (train_acc={best_acc*100:.1f}%)")
    
    # Apply to test set
    diff = scores[:, 1] - scores[:, 0]
    y_pred_qcnn = (diff > best_thr).astype(int)
    y_prob_qcnn = np.hstack([1-scores[:,[1]], scores[:,[1]]]) 
else:
    from scipy.special import softmax
    y_pred_qcnn = np.argmax(scores, axis=1)
    y_prob_qcnn = softmax(scores, axis=1)

acc_qcnn = accuracy_score(y_test_e, y_pred_qcnn)
print(f"\n  QCNN Accuracy: {acc_qcnn*100:.2f}%")
print(classification_report(y_test_e, y_pred_qcnn, target_names=le.classes_, zero_division=0))

ev.plot_combined(y_test_e, y_pred_qcnn, y_prob_qcnn, title="QCNN_Research")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — LOSS CURVES & SUMMARY
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
if n_classes == 1: axes = [axes]
for i, (loss, ax) in enumerate(zip(ovr_losses, axes)):
    ax.plot(loss, color='blue', marker='.')
    ax.set_title(le.classes_[i])
    ax.set_ylim(0, max(loss)*1.1 if loss else 1)
plt.tight_layout()
plt.savefig(f"{args.results}/qcnn_loss_curves.png")

joblib.dump({"params": ovr_params, "le": le, "pca": pca}, f"{args.results}/qcnn_model.joblib")

print("\n" + "="*60 + "\n  FINAL SUMMARY\n" + "="*60)
if acc_cnn: print(f"  {'Classical CNN':<20} Accuracy: {acc_cnn*100:>6.2f}% | Time: {cnn_time:>7.2f}s")
print(f"  {'Quantum QCNN':<20} Accuracy: {acc_qcnn*100:>6.2f}% | Time: {qcnn_time:>7.2f}s")
print(f"\n  Results saved to: {args.results}/")