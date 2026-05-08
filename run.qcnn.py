
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cirq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import argparse

from data.iot23_loader import IoT23Loader
from utils.evaluation import Evaluator

# ═══════════════════════════════════════════════════════════════
# STEP 0 — ARGUMENTS & CONFIG
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="QCNN Intrusion Detection — IoT-23")
parser.add_argument("--iot23",       required=True)
parser.add_argument("--n-samples",   type=int,   default=2000)
parser.add_argument("--n-qubits",    type=int,   default=4)
parser.add_argument("--n-layers",    type=int,   default=3)
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch-size",  type=int,   default=64)
parser.add_argument("--lr",          type=float, default=0.01)
parser.add_argument("--binary",      action="store_true")
parser.add_argument("--results",     default="results_qcnn")
args = parser.parse_args()

os.makedirs(args.results, exist_ok=True)
print("=" * 60)
print("  QCNN Intrusion Detection — IoT-23 (Unified Multi-Class)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("\n  STEP 1: Loading IoT-23 Dataset")
loader = IoT23Loader(args.iot23, max_rows_per_capture=100000)
X_raw, y_raw = loader.load_all()
print(f"  Loaded: {X_raw.shape[0]} flows × {X_raw.shape[1]} features")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — CLEANING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════
df = pd.DataFrame(X_raw)
df['label'] = y_raw
df = df.dropna()
X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

# ═══════════════════════════════════════════════════════════════
# STEP 3 — BALANCED SAMPLING & SPLITTING
# ═══════════════════════════════════════════════════════════════
print(f"\n  STEP 3: Balanced Sampling (Target: {args.n_samples})")
counts = pd.Series(y_encoded).value_counts()
target_per_class = args.n_samples // n_classes
indices = []
for cls in range(n_classes):
    cls_idx = np.where(y_encoded == cls)[0]
    n_pick = min(len(cls_idx), target_per_class)
    indices.extend(np.random.choice(cls_idx, n_pick, replace=False))

X_sampled = X[indices]
y_sampled = y_encoded[indices]

X_train, X_test, y_train_e, y_test_e = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# PCA for Quantum (Dense Encoding 3:1)
n_pca = args.n_qubits * 3
pca = PCA(n_components=min(n_pca, X_train.shape[1]))
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

print(f"  Final Train Size: {len(X_train_pca)} | PCA Variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — CLASSICAL CNN BASELINE
# ═══════════════════════════════════════════════════════════════
print("\n  STEP 4: Classical CNN Baseline")
ev = Evaluator(args.results)
acc_cnn = 0.0
cnn_time = 0.0

try:
    import tensorflow as tf
    def build_cnn(input_dim, n_cls):
        inp = tf.keras.Input(shape=(input_dim, 1))
        x = tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same")(inp)
        x = tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(256, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        out = tf.keras.layers.Dense(n_cls, activation="softmax")(x)
        mdl = tf.keras.Model(inp, out)
        mdl.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return mdl

    cnn_model = build_cnn(X_train_s.shape[1], n_classes)
    t0 = time.time()
    cnn_model.fit(X_train_s[:,:,None], y_train_e, epochs=50, batch_size=32, verbose=0)
    cnn_time = time.time() - t0
    y_prob_cnn = cnn_model.predict(X_test_s[:,:,None], verbose=0)
    y_pred_cnn = np.argmax(y_prob_cnn, axis=1)
    acc_cnn = accuracy_score(y_test_e, y_pred_cnn)
    print(f"  CNN Accuracy : {acc_cnn*100:.2f}%")
    ev.plot_combined(y_test_e, y_pred_cnn, y_prob_cnn, title="CNN_IoT23")
except Exception as e:
    print(f"  CNN Baseline failed: {e}")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — UNIFIED QCNN
# ═══════════════════════════════════════════════════════════════
print("\n  STEP 5: Unified Multi-Class QCNN Training")

def encode_input(x, n_q):
    qubits = cirq.LineQubit.range(n_q)
    c = cirq.Circuit()
    for i in range(n_q):
        f = (i*3) % len(x)
        c.append(cirq.rx(float(x[f]))(qubits[i]))
        c.append(cirq.ry(float(x[(f+1)%len(x)]))(qubits[i]))
        c.append(cirq.rz(float(x[(f+2)%len(x)]))(qubits[i]))
        if i < n_q - 1:
            c.append(cirq.ZZ(qubits[i], qubits[i+1])**(float(x[f])*0.1))
    return c, qubits

def conv_layer(qs, ps, off=0):
    c = cirq.Circuit()
    idx = off
    for i in range(0, len(qs)-1, 2):
        c.append(cirq.CNOT(qs[i], qs[i+1]))
        c.append(cirq.ry(ps[idx])(qs[i]))
        c.append(cirq.rz(ps[idx+1])(qs[i+1]))
        idx += 2
    return c, idx - off

def pool_layer(qs, ps, off=0):
    c = cirq.Circuit()
    rem = []
    idx = off
    for i in range(0, len(qs)-1, 2):
        c.append(cirq.CNOT(qs[i], qs[i+1]))
        c.append(cirq.ry(ps[idx])(qs[i+1]))
        rem.append(qs[i+1])
        idx += 1
    return c, (rem if rem else qs), idx - off

def build_qcnn_circuit(x, p, n_q, n_l):
    c, qs = encode_input(x, n_q)
    active = qs
    idx = 0
    for _ in range(n_l):
        ops, cons = conv_layer(active, p, idx)
        c.append(ops); idx += cons
        if len(active) > 1:
            ops, active, cons = pool_layer(active, p, idx)
            c.append(ops); idx += cons
    for q in active:
        c.append(cirq.ry(p[idx])(q))
        idx += 1
    return c, active, idx

def get_n_params(n_q, n_l):
    total, active = 0, n_q
    for _ in range(n_l):
        total += (active // 2) * 2 # conv
        if active > 1:
            total += (active // 2) # pool
            active = (active // 2)
    total += active # final ry
    return total

_sim = cirq.Simulator()

def multi_class_loss(x_batch, y_batch, p, n_q, n_l):
    total_l = 0
    for x, y_t in zip(x_batch, y_batch):
        c, active, _ = build_qcnn_circuit(x, p, n_q, n_l)
        res = _sim.simulate(c)
        scores = []
        for i in range(min(n_q, n_classes)):
            q = cirq.LineQubit(i)
            z = np.real(cirq.Z(q).expectation_from_state_vector(res.final_state_vector, qubit_map={qubit: k for k, qubit in enumerate(cirq.LineQubit.range(n_q))}))
            scores.append(z)
        while len(scores) < n_classes: scores.append(0.0)
        probs = np.exp(np.array(scores)*2.5)
        probs /= np.sum(probs)
        total_l -= np.log(probs[int(y_t)] + 1e-9)
    return total_l / len(x_batch)

def compute_grads(x_batch, y_batch, p, n_q, n_l):
    grads = np.zeros_like(p)
    shift = np.pi/2
    for i in range(len(p)):
        p[i] += shift; lp = multi_class_loss(x_batch, y_batch, p, n_q, n_l)
        p[i] -= 2*shift; lm = multi_class_loss(x_batch, y_batch, p, n_q, n_l)
        p[i] += shift; grads[i] = (lp - lm)/2
    return grads

n_p = get_n_params(args.n_qubits, args.n_layers)
p = np.random.uniform(-np.pi, np.pi, n_p)
m, v, t_s = np.zeros(n_p), np.zeros(n_p), 0
t_q_start = time.time()

for epoch in range(args.epochs):
    perm = np.random.permutation(len(X_train_pca))
    for i in range(0, len(X_train_pca), args.batch_size):
        idx = perm[i:i+args.batch_size]
        g = compute_grads(X_train_pca[idx], y_train_e[idx], p, args.n_qubits, args.n_layers)
        t_s += 1
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g**2
        mh, vh = m/(1-0.9**t_s), v/(1-0.999**t_s)
        p -= args.lr * mh / (np.sqrt(vh) + 1e-8)
    
    l_cur = multi_class_loss(X_train_pca[:50], y_train_e[:50], p, args.n_qubits, args.n_layers)
    print(f"    Epoch {epoch+1:2d} | loss={l_cur:.4f}")

qcnn_time = time.time() - t_q_start

# ═══════════════════════════════════════════════════════════════
# STEP 6 — EVALUATION
# ═══════════════════════════════════════════════════════════════
q_probs = []
for x in X_test_pca:
    c, active, _ = build_qcnn_circuit(x, p, args.n_qubits, args.n_layers)
    res = _sim.simulate(c)
    scores = []
    for i in range(min(args.n_qubits, n_classes)):
        q = cirq.LineQubit(i)
        z_val = np.real(cirq.Z(q).expectation_from_state_vector(res.final_state_vector, qubit_map={qubit: k for k, qubit in enumerate(cirq.LineQubit.range(args.n_qubits))}))
        scores.append(z_val)
    while len(scores) < n_classes: scores.append(0.0)
    pr = np.exp(np.array(scores)*2.5); q_probs.append(pr / np.sum(pr))

y_prob_q = np.array(q_probs)
y_pred_q = np.argmax(y_prob_q, axis=1)
acc_q = accuracy_score(y_test_e, y_pred_q)

print(f"\n  QCNN Accuracy: {acc_q*100:.2f}%")
print(classification_report(y_test_e, y_pred_q, target_names=le.classes_, zero_division=0))
ev.plot_combined(y_test_e, y_pred_q, y_prob_q, title="QCNN_Research")

print("\n" + "="*60 + "\n  FINAL SUMMARY\n" + "="*60)
print(f"  CNN Accuracy: {acc_cnn*100:.2f}% | Time: {cnn_time:.2f}s")
print(f"  QCNN Accuracy: {acc_q*100:.2f}% | Time: {qcnn_time:.2f}s")
print(f"\n  Results saved to: {args.results}/")