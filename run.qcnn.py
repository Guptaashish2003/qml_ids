import os
import time
import numpy as np
import pandas as pd
import cirq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import argparse

from data.iot23_loader import IoT23Loader
from utils.evaluation import Evaluator

"""
RESEARCH IMPLEMENTATION:
Security intrusion detection using quantum machine learning techniques
Authors: Kalinin & Krundyshev (2023)
FIXED EDITION — proper multi-class readout, stable SPSA, higher capacity
"""

parser = argparse.ArgumentParser(description="QCNN Intrusion Detection — IoT-23")
parser.add_argument("--iot23",       required=True)
parser.add_argument("--n-samples",   type=int,   default=2000)
parser.add_argument("--n-qubits",    type=int,   default=4)
parser.add_argument("--n-layers",    type=int,   default=3)
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch-size",  type=int,   default=64)
parser.add_argument("--lr",          type=float, default=0.01)
parser.add_argument("--results",     default="results_qcnn")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)
print("=" * 60)
print("  QCNN Intrusion Detection — IoT-23 (Fixed Edition)")
print("=" * 60)

# Evaluator is initialised after we know the class names (post label-encoding).
# We create a placeholder here and set .class_names later.
ev = None   # assigned after le.fit_transform below

# ═══════════════════════════════════════════════════════════════
# DATA PREPROCESSING  (unchanged from original)
# ═══════════════════════════════════════════════════════════════
loader = IoT23Loader(args.iot23, max_rows_per_capture=100000)
X_raw, y_raw = loader.load_all()
df = pd.DataFrame(X_raw); df['label'] = y_raw; df = df.dropna()
X, y = df.drop('label', axis=1).values, df['label'].values
le = LabelEncoder(); y_encoded = le.fit_transform(y); n_classes = len(le.classes_)

# Now we know the class names — initialise Evaluator
ev = Evaluator(class_names=list(le.classes_), save_dir=args.results)

# Stratified sampling — same as before
indices = []
target_per_class = args.n_samples // n_classes
for cls in range(n_classes):
    cls_idx = np.where(y_encoded == cls)[0]
    if len(cls_idx) > 0:
        indices.extend(
            np.random.choice(cls_idx, min(len(cls_idx), target_per_class), replace=False)
        )
X_s, y_s = X[indices], y_encoded[indices]
X_train, X_test, y_train_e, y_test_e = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42
)

# Scale → [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# FIX 1: use n_qubits features (not 6×) so re-uploading inside the circuit
# is the only source of repetition — keeps PCA components interpretable
n_features   = args.n_qubits          # one feature per qubit for clean angle encoding
pca          = PCA(n_components=min(n_features, X_train_s.shape[1]))
X_train_pca  = pca.fit_transform(X_train_s)
X_test_pca   = pca.transform(X_test_s)
# Clip to [0, π] after PCA (PCA can produce values outside scaler range)
X_train_pca  = np.clip(X_train_pca, 0, np.pi)
X_test_pca   = np.clip(X_test_pca,  0, np.pi)

# ═══════════════════════════════════════════════════════════════
# CNN BASELINE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  STEP 4: Classical CNN Baseline\n" + "="*60)
acc_cnn, cnn_time = 0.0, 0.0
y_prob_cnn = None   # (N_test, n_classes) — collected for Evaluator
try:
    import tensorflow as tf
    mdl = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
        tf.keras.layers.Conv1D(64, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])
    mdl.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    t0 = time.time()
    mdl.fit(X_train[:, :, None], y_train_e, epochs=20, verbose=0)
    cnn_time = time.time() - t0
    y_prob_cnn  = mdl.predict(X_test[:, :, None], verbose=0)   # (N, n_classes)
    y_pred_cnn  = np.argmax(y_prob_cnn, axis=1)
    acc_cnn     = accuracy_score(y_test_e, y_pred_cnn)
    print(f"  CNN Accuracy : {acc_cnn*100:.2f}%")
    print(classification_report(y_test_e, y_pred_cnn,
                                target_names=le.classes_, zero_division=0))

    # ── Evaluator: CNN confusion + ROC + combined plot ──────────
    ev.plot_confusion(y_test_e, y_pred_cnn,  title="Classical_CNN")
    ev.plot_roc(y_test_e, y_prob_cnn,         title="Classical_CNN")
    ev.plot_combined(y_test_e, y_pred_cnn, y_prob_cnn, title="Classical_CNN")

except Exception as e:
    print(f"  CNN failed: {e}")

# ═══════════════════════════════════════════════════════════════
# FIXED HIGH-CAPACITY QCNN
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  STEP 5: High-Capacity QCNN Training\n" + "="*60)

def build_qcnn_circuit(x, p, n_q, n_l):
    """
    FIX 2: Data re-uploading happens BETWEEN every conv layer so gradients
    receive a fresh data signal at each depth level.  The original code only
    encoded once at the top — this caused the gradient to vanish for deeper
    parameters.

    Parameter layout (per layer):
      • Re-upload  : n_q × 3 rotations  (Rx, Ry, Rz per qubit)
      • Conv pairs : (n_q//2) × 6 rotations + 1 CNOT per pair
    Total per layer: n_q*3 + (n_q//2)*6
    """
    qubits = cirq.LineQubit.range(n_q)
    c      = cirq.Circuit()
    n_enc  = n_q * 3                  # params used by re-upload block
    n_conv = (n_q // 2) * 6          # params used by conv block
    stride = n_enc + n_conv           # params per layer

    for layer in range(n_l):
        base = layer * stride

        # ── Data re-upload (3 rotations per qubit) ──────────────
        for i, q in enumerate(qubits):
            f_idx = i % len(x)
            c.append(cirq.rx(x[f_idx]           )(q))
            c.append(cirq.ry(p[base + i*3    ]  )(q))
            c.append(cirq.rz(p[base + i*3 + 1]  )(q))
            # second data re-upload angle (wrap around feature list)
            c.append(cirq.rx(x[(f_idx + 1) % len(x)])(q))
            c.append(cirq.rz(p[base + i*3 + 2]  )(q))

        # ── Conv block (6 learned params per qubit pair + CNOT) ─
        conv_base = base + n_enc
        pair_idx  = 0
        for i in range(0, n_q - 1, 2):
            q1, q2 = qubits[i], qubits[i + 1]
            pb = conv_base + pair_idx * 6
            c.append(cirq.rx(p[pb    ])(q1))
            c.append(cirq.ry(p[pb + 1])(q1))
            c.append(cirq.rz(p[pb + 2])(q1))
            c.append(cirq.rx(p[pb + 3])(q2))
            c.append(cirq.ry(p[pb + 4])(q2))
            c.append(cirq.rz(p[pb + 5])(q2))
            c.append(cirq.CNOT(q1, q2))
            pair_idx += 1

    return c, qubits


# FIX 3: Proper multi-class readout via one Pauli-Z expectation per class qubit.
# With 4 qubits we get 4 independent readout signals; we use a small linear
# head (learned weight matrix W_out) to map those 4 signals → 7 class logits.
# This completely replaces the broken "chunk the state vector" approach.

def get_expectations(x, p, n_q, n_l):
    """
    Returns a length-n_q vector of ⟨Z_i⟩ values, one per qubit.
    Each value is in [-1, 1].
    """
    c, qubits = build_qcnn_circuit(x, p, n_q, n_l)
    sim    = cirq.Simulator()
    result = sim.simulate(c)
    sv     = result.final_state_vector
    probs  = np.abs(sv) ** 2          # shape: (2**n_q,)
    n_total = 2 ** n_q

    exp_z = np.zeros(n_q)
    for qi in range(n_q):
        for state_idx in range(n_total):
            bit = (state_idx >> (n_q - 1 - qi)) & 1
            exp_z[qi] += probs[state_idx] * (1 - 2 * bit)
    return exp_z                      # shape: (n_q,)


# ── Parameter counts ────────────────────────────────────────────
n_enc_per_layer  = args.n_qubits * 3
n_conv_per_layer = (args.n_qubits // 2) * 6
n_circuit_params = (n_enc_per_layer + n_conv_per_layer) * args.n_layers

# FIX 4: Linear output head W_out maps n_qubits → n_classes (no bias needed
# because ⟨Z⟩ is already zero-centred).  Stored as a flat array appended to p.
n_head_params    = args.n_qubits * n_classes
n_total_params   = n_circuit_params + n_head_params

p   = np.random.uniform(-np.pi, np.pi, n_total_params)
# Initialise head with small weights so softmax starts near uniform
p[n_circuit_params:] = np.random.randn(n_head_params) * 0.1

print(f"  Circuit params : {n_circuit_params}")
print(f"  Head params    : {n_head_params}")
print(f"  Total params   : {n_total_params}")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def forward(x, p, n_q, n_l, n_cls):
    """Returns class probabilities via quantum expectations + linear head."""
    exp_z  = get_expectations(x, p[:n_circuit_params], n_q, n_l)
    W      = p[n_circuit_params:].reshape(n_cls, n_q)
    logits = W @ exp_z               # shape: (n_cls,)
    return softmax(logits)


def loss_fn(x_b, y_b, p, n_q, n_l, n_cls):
    """Mean cross-entropy over a mini-batch."""
    total = 0.0
    for x, yt in zip(x_b, y_b):
        probs  = forward(x, p, n_q, n_l, n_cls)
        total -= np.log(probs[int(yt)] + 1e-9)
    return total / len(x_b)


# FIX 5: Stable SPSA with proper Robbins-Monro constants.
# Original used ck decaying as (it+1)^{-0.101} which becomes ~0.07 by step 100
# — far too small for meaningful gradient signal.  Standard SPSA uses:
#   a_k = a / (A + k)^alpha,  c_k = c / k^gamma
# with alpha=0.602, gamma=0.101, A = 10% of total steps.
def spsa_grad(x_b, y_b, p, n_q, n_l, n_cls, it, c=0.2):
    """
    SPSA gradient estimate with fixed perturbation scale c.
    Using a fixed c (rather than a decaying schedule) keeps the gradient
    signal strong throughout training; the Adam update then handles scaling.
    """
    delta  = np.random.choice([-1, 1], size=len(p))
    l_p    = loss_fn(x_b, y_b, p + c * delta, n_q, n_l, n_cls)
    l_m    = loss_fn(x_b, y_b, p - c * delta, n_q, n_l, n_cls)
    return (l_p - l_m) / (2 * c * delta)


# ── Adam optimiser state ─────────────────────────────────────────
t_s = 0
m   = np.zeros(n_total_params)
v   = np.zeros(n_total_params)

# FIX 6: Use separate, larger lr for the classical head — it needs to move
# faster than the quantum circuit params to bootstrap useful gradients early.
lr_circuit = args.lr
lr_head    = args.lr * 5.0

t_start = time.time()

for epoch in range(args.epochs):
    perm = np.random.permutation(len(X_train_pca))
    for i in range(0, len(X_train_pca), args.batch_size):
        idx = perm[i : i + args.batch_size]
        g   = spsa_grad(
            X_train_pca[idx], y_train_e[idx],
            p, args.n_qubits, args.n_layers, n_classes, t_s
        )
        t_s += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t_s)
        v_hat = v / (1 - beta2 ** t_s)
        step  = m_hat / (np.sqrt(v_hat) + eps)

        # Apply separate learning rates to circuit vs head
        p[:n_circuit_params] -= lr_circuit * step[:n_circuit_params]
        p[n_circuit_params:] -= lr_head    * step[n_circuit_params:]

    l = loss_fn(
        X_train_pca[:50], y_train_e[:50],
        p, args.n_qubits, args.n_layers, n_classes
    )
    bar = '█' * int((1 - min(l / 3, 1)) * 20)
    print(f"    Epoch {epoch+1:2d}/{args.epochs} | loss={l:.4f} | {bar}")

q_time = time.time() - t_start

# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  STEP 6: Evaluation & Plots\n" + "="*60)

# Collect per-class probabilities for every test sample
y_p_q    = np.array([
    forward(x, p, args.n_qubits, args.n_layers, n_classes)
    for x in X_test_pca
])                                      # shape: (N_test, n_classes)
y_pred_q = np.argmax(y_p_q, axis=1)
acc_q    = accuracy_score(y_test_e, y_pred_q)

print(f"\n  QCNN Accuracy: {acc_q*100:.2f}%")
print(classification_report(
    y_test_e, y_pred_q,
    target_names=le.classes_,
    zero_division=0
))

# ── Evaluator: QCNN confusion matrix ────────────────────────────
ev.plot_confusion(y_test_e, y_pred_q, title="QCNN")

# ── Evaluator: QCNN ROC curves ──────────────────────────────────
ev.plot_roc(y_test_e, y_p_q, title="QCNN")

# ── Evaluator: combined side-by-side figure (matches paper Fig 7/8) ─
ev.plot_combined(y_test_e, y_pred_q, y_p_q, title="QCNN")

# ── Evaluator: training-time table (Table 4 style) ──────────────
timing_results = {}
if cnn_time > 0:
    timing_results["Classical CNN"] = [cnn_time]
timing_results["Quantum QCNN"]      = [q_time]
sizes_used = [len(X_train)]
Evaluator.print_table4(timing_results, sizes_used)

print("\n" + "="*60 + "\n  FINAL SUMMARY\n" + "="*60)
print(f"  Classical CNN        Accuracy:  {acc_cnn*100:0.2f}% | Time: {cnn_time:>7.2f}s")
print(f"  Quantum QCNN         Accuracy:  {acc_q*100:0.2f}% | Time: {q_time:>7.2f}s")
print(f"\n  Plots saved to → {args.results}/")