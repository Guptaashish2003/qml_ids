"""
Quantum Convolutional Neural Network (QCNN) Intrusion Detector
==============================================================
Implements Section 3.2 of the paper using TensorFlow Quantum (TFQ) + Cirq.

Architecture (mirrors Figure 2 of the paper)
---------------------------------------------
  Input qubits → [Convolution U_i] → [Sub-sampling V_j / measure] → ... → F → output

  - Convolution layer  : two-qubit parameterised unitary (Rx, Rz, CNOT)
  - Sub-sampling layer : measure a fraction of qubits; remaining qubits
                         get conditioned rotations (approximated here by
                         a classical pooling of measurement outcomes)
  - Fully-connected    : final unitary F on remaining qubits
  - Output             : expectation value of Z on readout qubit → sigmoid

Two backends
------------
  "tfq"     – TensorFlow Quantum (requires tensorflow-quantum)
  "cirq_sim"– Pure Cirq VQC simulation (slower, no TF dependency)

Classical CNN baseline is also provided for direct comparison (Table 4).

Usage
-----
    from models.qcnn_model import QCNNDetector, CNNBaseline
    model = QCNNDetector(n_qubits=8, n_layers=2, backend="cirq_sim")
    model.fit(X_train, y_train, epochs=20)
    preds = model.predict(X_test)
"""

import numpy as np
import time
from typing import Literal

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Classical CNN baseline
# ---------------------------------------------------------------------------
class CNNBaseline:
    """
    1-D CNN on flattened stream features — classical comparison baseline.
    Requires TensorFlow/Keras.
    """

    def __init__(self, n_classes: int = 7, filters: int = 32, epochs: int = 20):
        self.n_classes  = n_classes
        self.filters    = filters
        self.epochs     = epochs
        self.scaler     = MinMaxScaler()
        self.le         = LabelEncoder()
        self.model      = None
        self.train_time = None

    def _build(self, input_dim: int, n_classes: int):
        import tensorflow as tf
        inp  = tf.keras.Input(shape=(input_dim, 1))
        x    = tf.keras.layers.Conv1D(self.filters, 3, activation="relu", padding="same")(inp)
        x    = tf.keras.layers.MaxPooling1D(2)(x)
        x    = tf.keras.layers.Conv1D(self.filters * 2, 3, activation="relu", padding="same")(x)
        x    = tf.keras.layers.GlobalAveragePooling1D()(x)
        x    = tf.keras.layers.Dense(64, activation="relu")(x)
        out  = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        mdl  = tf.keras.Model(inp, out)
        mdl.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        return mdl

    def fit(self, X: np.ndarray, y, validation_split: float = 0.1) -> "CNNBaseline":
        import tensorflow as tf
        X_s = self.scaler.fit_transform(X)
        y_e = self.le.fit_transform(y)
        X_s = X_s[:, :, np.newaxis]                 # (N, features, 1)
        self.model = self._build(X.shape[1], len(self.le.classes_))
        t0 = time.time()
        self.model.fit(X_s, y_e,
                       epochs=self.epochs,
                       batch_size=64,
                       validation_split=validation_split,
                       verbose=1)
        self.train_time = time.time() - t0
        print(f"[CNN] Training time: {self.train_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)[:, :, np.newaxis]
        probs = self.model.predict(X_s, verbose=0)
        return np.argmax(probs, axis=1)

    def report(self, X_test, y_test) -> str:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return classification_report(y_true, y_pred,
                                     target_names=self.le.classes_,
                                     zero_division=0)


# ---------------------------------------------------------------------------
# QCNN layer builders (Cirq)
# ---------------------------------------------------------------------------
def _conv_layer(qubits, params, layer_idx: int):
    """
    Parameterised two-qubit convolutional unitary U_i.
    Applied in a brick-wall pattern to adjacent qubit pairs.
    """
    import cirq
    ops = []
    offset = layer_idx % 2
    for i in range(offset, len(qubits) - 1, 2):
        q0, q1 = qubits[i], qubits[i + 1]
        pi = layer_idx * (len(qubits) - 1) * 3 + i * 3
        ops += [
            cirq.rx(params[pi    ])(q0),
            cirq.rz(params[pi + 1])(q1),
            cirq.CNOT(q0, q1),
            cirq.ry(params[pi + 2])(q1),
        ]
    return ops


def _pool_layer(source_qubits, sink_qubits, params, layer_idx: int):
    """
    Sub-sampling: measure source qubits; conditioned Rz on sink qubits.
    In a VQC simulation we approximate this with parameterised Rz gates.
    """
    import cirq
    ops = []
    for k, (src, snk) in enumerate(zip(source_qubits, sink_qubits)):
        pi = layer_idx * len(source_qubits) + k
        ops += [cirq.CNOT(src, snk), cirq.rz(params[pi])(snk)]
    return ops


def _build_qcnn_circuit(n_qubits: int, n_layers: int, params: np.ndarray):
    """
    Build a full QCNN circuit:
        n_layers × (conv + pool), then a fully-connected unitary F.
    """
    import cirq
    qubits  = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    active  = list(qubits)
    idx     = 0

    # Encoding layer: Rx on every qubit
    for q in active:
        circuit.append(cirq.H(q))

    n_conv_params = n_layers * (len(active) - 1) * 3
    n_pool_params = n_layers * (len(active) // 2)
    expected      = n_conv_params + n_pool_params + len(active)

    for layer in range(n_layers):
        # Convolution
        circuit.append(_conv_layer(active, params[idx:], layer))
        idx += (len(active) - 1) * 3

        # Pooling: measure even qubits, condition on odd
        sources = active[::2]
        sinks   = active[1::2]
        circuit.append(_pool_layer(sources, sinks, params[idx:], layer))
        idx    += len(sources)
        active  = sinks   # half the active qubits after each pool step

    # Fully-connected unitary F on remaining qubits
    for k, q in enumerate(active):
        circuit.append(cirq.rz(params[idx + k])(q))

    return circuit, active


# ---------------------------------------------------------------------------
# QCNN Detector
# ---------------------------------------------------------------------------
class QCNNDetector:
    """
    QCNN-based intrusion detector.

    Parameters
    ----------
    n_qubits  : number of qubits (must be 2^k for clean pooling)
    n_layers  : number of conv+pool layer pairs
    backend   : "tfq" (TensorFlow Quantum) | "cirq_sim" (pure Cirq VQC)
    epochs    : number of training epochs
    lr        : learning rate
    batch_size: mini-batch size
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        backend: Literal["tfq", "cirq_sim"] = "cirq_sim",
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 32,
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

        # Compute total parameter count
        n  = n_qubits
        nc = sum((n >> i) - 1 for i in range(n_layers)) * 3
        np_ = sum((n >> i) // 2 for i in range(n_layers))
        nf  = n >> n_layers
        self.n_params   = nc + np_ + nf
        self.params     = np.random.uniform(-np.pi, np.pi, self.n_params)

    # ------------------------------------------------------------------
    def _encode_input(self, x: np.ndarray):
        """Angle-encode one feature vector onto qubits."""
        import cirq
        qubits  = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            val = float(x[i]) if i < len(x) else 0.0
            circuit.append(cirq.rx(val)(q))
        return circuit

    # ------------------------------------------------------------------
    def _forward(self, x: np.ndarray) -> float:
        """
        Forward pass (Cirq simulation):
          1. Encode x onto qubits
          2. Apply QCNN circuit with current params
          3. Measure Z-expectation on readout qubit
        Returns a float in [-1, 1].
        """
        import cirq
        enc_circuit            = self._encode_input(x)
        qcnn_circuit, active   = _build_qcnn_circuit(
            self.n_qubits, self.n_layers, self.params)
        full_circuit           = enc_circuit + qcnn_circuit

        sim    = cirq.Simulator()
        result = sim.simulate(full_circuit)
        sv     = result.final_state_vector

        # Compute ⟨Z⟩ on the readout qubit (last active qubit)
        readout_idx = active[-1].x if hasattr(active[-1], "x") else 0
        n_total     = 2 ** self.n_qubits
        probs       = np.abs(sv) ** 2
        exp_z       = 0.0
        for state_idx in range(n_total):
            bit = (state_idx >> (self.n_qubits - 1 - readout_idx)) & 1
            exp_z += probs[state_idx] * (1 - 2 * bit)
        return exp_z

    # ------------------------------------------------------------------
    def _parameter_shift(
        self, x: np.ndarray, y: float, param_idx: int, eps: float = np.pi / 2
    ) -> float:
        """Parameter-shift rule gradient for one parameter."""
        orig = self.params[param_idx]
        self.params[param_idx] = orig + eps
        f_plus  = self._forward(x)
        self.params[param_idx] = orig - eps
        f_minus = self._forward(x)
        self.params[param_idx] = orig
        # dL/dθ = (f+ - f-) / 2  (MSE derivative)
        return (f_plus - f_minus) / 2 * (f_plus + f_minus) / 2 - y

    # ------------------------------------------------------------------
    def _fit_cirq(self, X: np.ndarray, y_binary: np.ndarray):
        """
        Train binary QCNN via parameter-shift gradient descent.
        For multi-class: one-vs-rest scheme is applied by fit().
        """
        rng = np.random.default_rng(42)
        for epoch in range(self.epochs):
            total_loss = 0.0
            idx_perm   = rng.permutation(len(X))
            for i in idx_perm[:self.batch_size]:   # mini-batch
                pred = self._forward(X[i])
                loss = 0.5 * (pred - y_binary[i]) ** 2
                total_loss += loss
                grads = np.array([
                    self._parameter_shift(X[i], y_binary[i], k)
                    for k in range(self.n_params)
                ])
                self.params -= self.lr * grads
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss/self.batch_size:.4f}")

    # ------------------------------------------------------------------
    def _fit_tfq(self, X: np.ndarray, y: np.ndarray):
        """Train QCNN using TensorFlow Quantum + Keras."""
        import cirq, tensorflow as tf, tensorflow_quantum as tfq

        qubits = cirq.LineQubit.range(self.n_qubits)
        # Build parameterised model circuit
        import sympy
        symbols       = sympy.symbols(f"θ0:{self.n_params}")
        params_tensor = tf.Variable(
            np.random.uniform(-np.pi, np.pi, self.n_params).astype(np.float32)
        )
        # Data input circuits
        circuits_tensor = tfq.convert_to_tensor(
            [self._encode_input(x) for x in X])

        qcnn_circuit, active = _build_qcnn_circuit(
            self.n_qubits, self.n_layers, np.zeros(self.n_params))

        readout_op = cirq.Z(active[-1])
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
        X_s = self.scaler.fit_transform(X)
        y_e = self.le.fit_transform(y)
        # Reduce to n_qubits features
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=self.n_qubits)
        X_r = self._pca.fit_transform(X_s)
        X_r = np.clip(X_r, 0, np.pi)

        n_classes = len(self.le.classes_)
        t0 = time.time()

        if self.backend == "tfq":
            try:
                # Binary-ify for TFQ demo (multi-class = one-vs-rest)
                self._fit_tfq(X_r, (y_e == 0).astype(float))
            except ImportError as e:
                print(f"[QCNN] TFQ unavailable ({e}), falling back to cirq_sim.")
                self.backend = "cirq_sim"
                return self.fit(X, y)

        elif self.backend == "cirq_sim":
            # One-vs-rest classifiers
            self._ovr_params = []
            for cls_idx in range(n_classes):
                print(f"\n[QCNN] Training class {cls_idx}/{n_classes-1} "
                      f"({self.le.classes_[cls_idx]}) …")
                y_bin = np.where(y_e == cls_idx, 1.0, -1.0)
                self.params = np.random.uniform(-np.pi, np.pi, self.n_params)
                self._fit_cirq(X_r, y_bin)
                self._ovr_params.append(self.params.copy())

        self.train_time = time.time() - t0
        self._X_r_train = X_r
        print(f"\n[QCNN/{self.backend}] Training time: {self.train_time:.2f}s")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        X_r = np.clip(self._pca.transform(X_s), 0, np.pi)
        scores = np.zeros((len(X_r), len(self.le.classes_)))

        for cls_idx, p in enumerate(self._ovr_params):
            self.params = p
            scores[:, cls_idx] = [self._forward(x) for x in X_r]

        return np.argmax(scores, axis=1)

    def report(self, X_test, y_test) -> str:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return classification_report(y_true, y_pred,
                                     target_names=self.le.classes_,
                                     zero_division=0)

    def confusion(self, X_test, y_test) -> np.ndarray:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.stream_builder import StreamBuilder

    print("Generating synthetic dataset …")
    df = StreamBuilder.generate_synthetic(n_streams=200)
    feat_cols = [c for c in df.columns if c != "label"]
    X = df[feat_cols].values
    y = df["label"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n--- Classical CNN baseline ---")
    cnn = CNNBaseline(epochs=5)
    cnn.fit(X_train, y_train)
    print(cnn.report(X_test, y_test))

    print("\n--- QCNN (Cirq sim, 4 qubits, 1 layer) ---")
    qcnn = QCNNDetector(n_qubits=4, n_layers=1, backend="cirq_sim",
                        epochs=5, batch_size=16)
    qcnn.fit(X_train, y_train)
    print(qcnn.report(X_test, y_test))
