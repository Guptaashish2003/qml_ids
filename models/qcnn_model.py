"""
Quantum Convolutional Neural Network (QCNN) Intrusion Detector
==============================================================
Implements Section 3.2 of the paper using Cirq.

Architecture (mirrors Figure 2 of the paper)
---------------------------------------------
  Input qubits → [Convolution U_i] → [Sub-sampling V_j / measure] → ... → F → output

  - Convolution layer  : two-qubit parameterised unitary (Rx, Rz, CNOT, Ry)
                         applied in brick-wall pattern
  - Sub-sampling layer : CNOT(src→snk) + Rz(snk) — approximates conditioned
                         rotation after partial measurement
  - Fully-connected    : Rz on every remaining qubit
  - Output             : ⟨Z⟩ expectation on the last remaining qubit

FIX NOTES vs original
---------------------
1. Pooling is capped so we never collapse below 2 active qubits.
   With n_qubits=4, n_layers=2 the original pooled to 1 qubit — a single
   ⟨Z⟩ scalar can't separate 7 classes in one-vs-rest.
2. n_params is recalculated to match the capped pooling depth.
3. CNNBaseline exposes `predict_proba` so run_qcnn.py can build ROC curves
   without reaching into Keras internals.
4. QCNNDetector exposes `predict_proba` for the same reason.

Backends
--------
  "cirq_sim" — Pure Cirq VQC simulation (default, no TF dependency)
  "tfq"      — TensorFlow Quantum (requires tensorflow-quantum)
"""

import numpy as np
import time
from typing import Literal

import cirq
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Classical CNN baseline
# ─────────────────────────────────────────────────────────────────────────────
class CNNBaseline:
    """
    1-D CNN on flattened stream features — classical comparison baseline.
    Requires TensorFlow / Keras.
    """

    def __init__(self, n_classes: int = 7, filters: int = 64, epochs: int = 20):
        self.n_classes  = n_classes
        self.filters    = filters
        self.epochs     = epochs
        self.scaler     = MinMaxScaler()
        self.le         = LabelEncoder()
        self.model      = None
        self.train_time = None

    # ------------------------------------------------------------------
    def _build(self, input_dim: int, n_classes: int):
        import tensorflow as tf
        inp = tf.keras.Input(shape=(input_dim, 1))
        x   = tf.keras.layers.Conv1D(self.filters, 3, activation="relu", padding="same")(inp)
        x   = tf.keras.layers.MaxPooling1D(2)(x)
        x   = tf.keras.layers.Conv1D(self.filters * 2, 3, activation="relu", padding="same")(x)
        x   = tf.keras.layers.GlobalAveragePooling1D()(x)
        x   = tf.keras.layers.Dense(64, activation="relu")(x)
        out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        mdl = tf.keras.Model(inp, out)
        mdl.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        return mdl

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y, validation_split: float = 0.1) -> "CNNBaseline":
        X_s = self.scaler.fit_transform(X)
        y_e = self.le.fit_transform(y)
        X_s = X_s[:, :, np.newaxis]
        self.model = self._build(X.shape[1], len(self.le.classes_))
        t0 = time.time()
        self.model.fit(X_s, y_e,
                       epochs=self.epochs,
                       batch_size=64,
                       validation_split=validation_split,
                       verbose=0)
        self.train_time = time.time() - t0
        print(f"[CNN] Training time: {self.train_time:.2f}s")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, n_classes) softmax probability matrix."""
        X_s = self.scaler.transform(X)[:, :, np.newaxis]
        return self.model.predict(X_s, verbose=0)

    # ------------------------------------------------------------------
    def report(self, X_test, y_test) -> str:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return classification_report(y_true, y_pred,
                                     target_names=self.le.classes_,
                                     zero_division=0)


# ─────────────────────────────────────────────────────────────────────────────
# QCNN circuit builders
# ─────────────────────────────────────────────────────────────────────────────
def _conv_layer(qubits, params, layer_idx: int):
    """
    Brick-wall two-qubit conv unitary U_i.
    4 params per pair: Rx(q0), Rz(q1), CNOT, Ry(q1).
    Offset alternates between layers so every pair gets covered.
    """
    ops    = []
    offset = layer_idx % 2
    n      = len(qubits)
    for i in range(offset, n - 1, 2):
        q0, q1 = qubits[i], qubits[i + 1]
        # Index into the params slice passed in
        pi = i * 3           # 3 params per pair in this slice
        ops += [
            cirq.rx(params[pi    ])(q0),
            cirq.rz(params[pi + 1])(q1),
            cirq.CNOT(q0, q1),
            cirq.ry(params[pi + 2])(q1),
        ]
    return ops


def _pool_layer(sources, sinks, params):
    """
    Sub-sampling: CNOT(src→snk) + Rz(snk, param).
    One param per source-sink pair.
    """
    ops = []
    for k, (src, snk) in enumerate(zip(sources, sinks)):
        ops += [cirq.CNOT(src, snk), cirq.rz(params[k])(snk)]
    return ops


def _count_params(n_qubits: int, n_layers: int) -> int:
    """
    Count total circuit parameters respecting the 2-qubit floor on pooling.

    Per layer:
      conv  : ceil((active-1)/2) pairs × 3 params  (brick-wall, both offsets)
              We count both offsets (even layer + odd layer) = (active-1) pairs total
              → (active - 1) * 3
      pool  : active // 2 params  (one per source qubit)
    After pool: active = max(active // 2, 2)   ← FIX: floor at 2

    Fully-connected F: active_final params (one Rz per remaining qubit)
    """
    active = n_qubits
    total  = 0
    for _ in range(n_layers):
        total += (active - 1) * 3          # conv
        total += active // 2               # pool
        active = max(active // 2, 2)       # FIX: never collapse below 2
    total += active                        # fully-connected F
    return total


def _build_qcnn_circuit(n_qubits: int, n_layers: int, params: np.ndarray):
    """
    Build full QCNN circuit.
    Returns (circuit, active_qubits_list).
    """
    qubits  = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    active  = list(qubits)
    idx     = 0

    # Hadamard encoding layer
    for q in active:
        circuit.append(cirq.H(q))

    for layer in range(n_layers):
        n_active = len(active)

        # ── Convolution ─────────────────────────────────────────
        n_conv = (n_active - 1) * 3
        circuit.append(_conv_layer(active, params[idx:], layer))
        idx += n_conv

        # ── Sub-sampling ────────────────────────────────────────
        sources  = active[::2]
        sinks    = active[1::2]
        n_pool   = len(sources)
        circuit.append(_pool_layer(sources, sinks, params[idx:]))
        idx     += n_pool

        # FIX: floor active qubits at 2 so readout is always meaningful
        new_active = sinks if len(sinks) >= 2 else active[-2:]
        active     = new_active

    # ── Fully-connected F ────────────────────────────────────────
    for k, q in enumerate(active):
        circuit.append(cirq.rz(params[idx + k])(q))

    return circuit, active


# ─────────────────────────────────────────────────────────────────────────────
# QCNN Detector
# ─────────────────────────────────────────────────────────────────────────────
class QCNNDetector:
    """
    QCNN-based intrusion detector.

    Parameters
    ----------
    n_qubits   : number of qubits (power of 2 recommended)
    n_layers   : conv+pool layer pairs (pooling floors at 2 active qubits)
    backend    : "cirq_sim" | "tfq"
    epochs     : training epochs per OvR binary classifier
    lr         : Adam learning rate
    batch_size : samples per gradient step
    """

    def __init__(
        self,
        n_qubits   : int = 4,
        n_layers   : int = 2,
        backend    : Literal["tfq", "cirq_sim"] = "cirq_sim",
        epochs     : int  = 20,
        lr         : float = 0.01,
        batch_size : int  = 32,
    ):
        self.n_qubits   = n_qubits
        self.n_layers   = n_layers
        self.backend    = backend
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size

        self.scaler     = MinMaxScaler(feature_range=(0, np.pi))
        self.le         = LabelEncoder()
        self.train_time = None
        self._pca       = None
        self._ovr_params: list = []

        # Correct param count respecting the 2-qubit floor
        self.n_params = _count_params(n_qubits, n_layers)
        self.params   = np.random.uniform(-np.pi, np.pi, self.n_params)

    # ------------------------------------------------------------------
    def _encode_input(self, x: np.ndarray):
        """Angle-encode feature vector: one Rx per qubit."""
        qubits  = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            val = float(x[i]) if i < len(x) else 0.0
            circuit.append(cirq.rx(val)(q))
        return circuit

    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> float:
        """
        Forward pass: encode → QCNN → ⟨Z⟩ on last active qubit.
        Returns float in [-1, 1].
        """
        enc_circuit          = self._encode_input(x)
        qcnn_circuit, active = _build_qcnn_circuit(
            self.n_qubits, self.n_layers, self.params)
        full_circuit         = enc_circuit + qcnn_circuit

        sim    = cirq.Simulator()
        result = sim.simulate(full_circuit)
        sv     = result.final_state_vector
        probs  = np.abs(sv) ** 2

        # ⟨Z⟩ on the last active qubit
        readout_idx = active[-1].x
        n_total     = 2 ** self.n_qubits
        exp_z       = 0.0
        for state_idx in range(n_total):
            bit    = (state_idx >> (self.n_qubits - 1 - readout_idx)) & 1
            exp_z += probs[state_idx] * (1 - 2 * bit)
        return float(exp_z)

    # ------------------------------------------------------------------
    def _parameter_shift(self, x: np.ndarray, y: float, param_idx: int) -> float:
        """Parameter-shift rule gradient for one parameter (MSE loss)."""
        eps = np.pi / 2
        orig = self.params[param_idx]

        self.params[param_idx] = orig + eps
        f_plus = self._forward(x)

        self.params[param_idx] = orig - eps
        f_minus = self._forward(x)

        self.params[param_idx] = orig
        # d(MSE)/dθ  where MSE = 0.5*(f - y)²
        f_mid = (f_plus + f_minus) / 2
        return (f_plus - f_minus) / 2 * (f_mid - y)

    # ------------------------------------------------------------------
    def _fit_cirq(self, X: np.ndarray, y_binary: np.ndarray):
        """
        Train a binary OvR classifier via parameter-shift + SGD.
        y_binary ∈ {+1, -1}.
        """
        rng = np.random.default_rng(42)
        for epoch in range(self.epochs):
            total_loss = 0.0
            perm       = rng.permutation(len(X))
            batch_idx  = perm[:self.batch_size]
            for i in batch_idx:
                pred        = self._forward(X[i])
                loss        = 0.5 * (pred - y_binary[i]) ** 2
                total_loss += loss
                grads = np.array([
                    self._parameter_shift(X[i], y_binary[i], k)
                    for k in range(self.n_params)
                ])
                self.params -= self.lr * grads
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs} "
                      f"loss={total_loss/self.batch_size:.4f}")

    # ------------------------------------------------------------------
    def _fit_tfq(self, X: np.ndarray, y: np.ndarray):
        """Train QCNN using TensorFlow Quantum + Keras (binary OvR)."""
        import tensorflow as tf
        import tensorflow_quantum as tfq

        circuits_tensor      = tfq.convert_to_tensor(
            [self._encode_input(x) for x in X])
        qcnn_circuit, active = _build_qcnn_circuit(
            self.n_qubits, self.n_layers, np.zeros(self.n_params))
        readout_op           = cirq.Z(active[-1])

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(qcnn_circuit, readout_op),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                      loss="mse", metrics=["mae"])
        model.fit(circuits_tensor, y.astype(np.float32),
                  epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        self._tfq_model = model

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y) -> "QCNNDetector":
        # Preprocessing — shared scaler + PCA
        X_s = self.scaler.fit_transform(X)
        y_e = self.le.fit_transform(y)

        self._pca = PCA(n_components=self.n_qubits)
        X_r       = np.clip(self._pca.fit_transform(X_s), 0, np.pi)

        n_classes = len(self.le.classes_)
        t0        = time.time()

        if self.backend == "tfq":
            try:
                self._fit_tfq(X_r, (y_e == 0).astype(float))
            except ImportError as e:
                print(f"[QCNN] TFQ unavailable ({e}), falling back to cirq_sim.")
                self.backend = "cirq_sim"
                return self.fit(X, y)

        elif self.backend == "cirq_sim":
            self._ovr_params = []
            for cls_idx in range(n_classes):
                print(f"\n[QCNN] Training class {cls_idx+1}/{n_classes} "
                      f"— '{self.le.classes_[cls_idx]}'")
                y_bin = np.where(y_e == cls_idx, 1.0, -1.0)
                # Fresh random params for every binary classifier
                self.params = np.random.uniform(-np.pi, np.pi, self.n_params)
                self._fit_cirq(X_r, y_bin)
                self._ovr_params.append(self.params.copy())

        self.train_time = time.time() - t0
        print(f"\n[QCNN/{self.backend}] Training complete — {self.train_time:.2f}s")
        return self

    # ------------------------------------------------------------------
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, n_classes) matrix of raw ⟨Z⟩ scores in [-1, 1].
        Used internally by predict() and predict_proba().
        """
        X_s    = self.scaler.transform(X)
        X_r    = np.clip(self._pca.transform(X_s), 0, np.pi)
        scores = np.zeros((len(X_r), len(self.le.classes_)))
        for cls_idx, p in enumerate(self._ovr_params):
            self.params        = p
            scores[:, cls_idx] = [self._forward(x) for x in X_r]
        return scores

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns integer-encoded class predictions."""
        return np.argmax(self._raw_scores(X), axis=1)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, n_classes) probability matrix.
        Converts raw ⟨Z⟩ scores to [0,1] via row-wise min-max then L1 norm.
        """
        scores  = self._raw_scores(X)
        s_min   = scores.min(axis=1, keepdims=True)
        s_max   = scores.max(axis=1, keepdims=True)
        probs   = (scores - s_min) / (s_max - s_min + 1e-9)
        probs  /= probs.sum(axis=1, keepdims=True)
        return probs

    # ------------------------------------------------------------------
    def report(self, X_test, y_test) -> str:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return classification_report(y_true, y_pred,
                                     target_names=self.le.classes_,
                                     zero_division=0)

    # ------------------------------------------------------------------
    def confusion(self, X_test, y_test) -> np.ndarray:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return confusion_matrix(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.stream_builder import StreamBuilder

    print("Generating synthetic dataset …")
    df        = StreamBuilder.generate_synthetic(n_streams=200)
    feat_cols = [c for c in df.columns if c != "label"]
    X         = df[feat_cols].values
    y         = df["label"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n--- Classical CNN baseline ---")
    cnn = CNNBaseline(epochs=5)
    cnn.fit(X_train, y_train)
    print(cnn.report(X_test, y_test))

    print("\n--- QCNN (Cirq sim, 4 qubits, 2 layers) ---")
    qcnn = QCNNDetector(n_qubits=4, n_layers=2, backend="cirq_sim",
                        epochs=5, batch_size=16)
    print(f"  n_params = {qcnn.n_params}")
    qcnn.fit(X_train, y_train)
    print(qcnn.report(X_test, y_test))