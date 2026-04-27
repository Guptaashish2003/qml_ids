"""
Quantum Support Vector Machine (QSVM) Intrusion Detector
=========================================================
Implements Section 3.1 of the paper.

Two execution modes
-------------------
1. Qiskit  – uses qiskit-machine-learning's QSVM with a quantum kernel
             (ZZFeatureMap). Runs on Aer simulator or real IBM Q hardware.

2. Cirq/TFQ – uses a parameterised quantum circuit as a kernel estimator
              with TensorFlow Quantum (optional, falls back gracefully).

3. Classical SVM baseline – wrapped identically so comparisons are trivial.

Usage
-----
    from models.qsvm_model import QSVMDetector, SVMBaseline
    from sklearn.preprocessing import LabelEncoder

    model = QSVMDetector(backend="qiskit", n_qubits=4)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(model.report(X_test, y_test))
"""

import numpy as np
import time
from typing import Literal, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
class _BaseDetector:
    def __init__(self):
        self.scaler     = MinMaxScaler(feature_range=(0, np.pi))
        self.le         = LabelEncoder()
        self.model      = None
        self.train_time = None

    def _preprocess(self, X, fit=False):
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def report(self, X_test, y_test) -> str:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test) if hasattr(self.le, "classes_") else y_test
        return classification_report(y_true, y_pred,
                                     target_names=self.le.classes_,
                                     zero_division=0)

    def confusion(self, X_test, y_test) -> np.ndarray:
        y_pred = self.predict(X_test)
        y_true = self.le.transform(y_test)
        return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# 1. Classical SVM baseline (for benchmark comparison as in Table 4)
# ---------------------------------------------------------------------------
class SVMBaseline(_BaseDetector):
    """Conventional SVM intrusion detector — used as the comparison baseline."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0, **kwargs):
        super().__init__()
        self.model = SVC(kernel=kernel, C=C, probability=True, **kwargs)

    def fit(self, X: np.ndarray, y) -> "SVMBaseline":
        X_s = self._preprocess(X, fit=True)
        y_e = self.le.fit_transform(y)
        t0  = time.time()
        self.model.fit(X_s, y_e)
        self.train_time = time.time() - t0
        print(f"[SVM] Training time: {self.train_time:.2f}s  |  "
              f"classes: {list(self.le.classes_)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._preprocess(X))


# ---------------------------------------------------------------------------
# 2. QSVM – Qiskit backend
# ---------------------------------------------------------------------------
class QSVMDetector(_BaseDetector):
    """
    QSVM intrusion detector using Qiskit's FidelityQuantumKernel.

    The quantum kernel is computed via:
        K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
    where |φ(x)⟩ is the state prepared by ZZFeatureMap(x).

    Parameters
    ----------
    backend  : "qiskit" | "cirq_sim"
    n_qubits : Number of qubits / feature dimensions after PCA reduction.
               Paper uses full features; we default to 4 for simulator speed.
    reps     : ZZFeatureMap repetitions (depth of entanglement).
    C        : SVC regularisation parameter.
    """

    def __init__(
        self,
        backend: Literal["qiskit", "cirq_sim"] = "qiskit",
        n_qubits: int = 4,
        reps: int = 2,
        C: float = 1.0,
    ):
        super().__init__()
        self.backend  = backend
        self.n_qubits = n_qubits
        self.reps     = reps
        self.C        = C
        self._svc     = None
        self._kernel  = None

    # ------------------------------------------------------------------
    def _build_qiskit_kernel(self):
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        from qiskit.primitives import Sampler

        feature_map  = ZZFeatureMap(feature_dimension=self.n_qubits,
                                    reps=self.reps)
        sampler      = Sampler()
        fidelity     = ComputeUncompute(sampler=sampler)
        self._kernel = FidelityQuantumKernel(fidelity=fidelity,
                                             feature_map=feature_map)
        return self._kernel

    # ------------------------------------------------------------------
    def _build_cirq_kernel(self):
        """
        A lightweight Cirq-based kernel: inner product of statevectors
        produced by angle-encoding circuits (fallback when TFQ unavailable).

        Vectorised: precompute all statevectors, then K = |SV1 · SV2†|².
        """
        import cirq

        sim = cirq.Simulator()
        n_q = self.n_qubits
        qubits = cirq.LineQubit.range(n_q)

        def _statevectors(X: np.ndarray) -> np.ndarray:
            """Return (len(X), 2**n_qubits) complex matrix of statevectors."""
            svs = []
            for x in X:
                c = cirq.Circuit()
                for i, val in enumerate(x[:n_q]):
                    c.append(cirq.rx(float(val))(qubits[i]))
                result = sim.simulate(c)
                svs.append(result.final_state_vector)
            return np.array(svs)

        def kernel_fn(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
            sv1 = _statevectors(X1)   # (n1, 2^nq)
            sv2 = _statevectors(X2)   # (n2, 2^nq)
            # K(i,j) = |<φ(x_i)|φ(x_j)>|²
            return np.abs(sv1 @ sv2.conj().T) ** 2

        return kernel_fn

    # ------------------------------------------------------------------
    def _reduce_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """PCA reduction to n_qubits dimensions for quantum kernel efficiency."""
        if not hasattr(self, "_pca"):
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.n_qubits)
        if fit:
            return self._pca.fit_transform(X)
        return self._pca.transform(X)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y) -> "QSVMDetector":
        X_s = self._preprocess(X, fit=True)
        X_r = self._reduce_features(X_s, fit=True)
        y_e = self.le.fit_transform(y)

        t0 = time.time()

        if self.backend == "qiskit":
            try:
                kernel = self._build_qiskit_kernel()
                K_train = kernel.evaluate(x_vec=X_r)
                self._svc = SVC(kernel="precomputed", C=self.C, probability=True)
                self._svc.fit(K_train, y_e)
                self._X_train = X_r
            except ImportError as e:
                print(f"[QSVM] Qiskit unavailable ({e}), falling back to Cirq sim.")
                self.backend = "cirq_sim"
                return self.fit(X, y)

        elif self.backend == "cirq_sim":
            self._cirq_kernel = self._build_cirq_kernel()
            K_train = self._cirq_kernel(X_r, X_r)
            self._svc = SVC(kernel="precomputed", C=self.C, probability=True)
            self._svc.fit(K_train, y_e)
            self._X_train = X_r

        self.train_time = time.time() - t0
        print(f"[QSVM/{self.backend}] Training time: {self.train_time:.2f}s  |  "
              f"n_qubits={self.n_qubits}  |  classes: {list(self.le.classes_)}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._preprocess(X)
        X_r = self._reduce_features(X_s)

        if self.backend == "qiskit":
            K_test = self._kernel.evaluate(x_vec=X_r, y_vec=self._X_train)
        else:
            K_test = self._cirq_kernel(X_r, self._X_train)

        return self._svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = self._preprocess(X)
        X_r = self._reduce_features(X_s)
        if self.backend == "qiskit":
            K_test = self._kernel.evaluate(x_vec=X_r, y_vec=self._X_train)
        else:
            K_test = self._cirq_kernel(X_r, self._X_train)
        return self._svc.predict_proba(K_test)

    # ------------------------------------------------------------------
    # Pickle support — rebuild the cirq kernel closure on load
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable closure; will be rebuilt in __setstate__
        state.pop("_cirq_kernel", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.backend == "cirq_sim":
            self._cirq_kernel = self._build_cirq_kernel()


# ---------------------------------------------------------------------------
# CLI demo / benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.stream_builder import StreamBuilder

    print("Generating synthetic dataset …")
    df = StreamBuilder.generate_synthetic(n_streams=500)
    feat_cols = [c for c in df.columns if c != "label"]
    X = df[feat_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("\n--- Classical SVM baseline ---")
    svm = SVMBaseline()
    svm.fit(X_train, y_train)
    print(svm.report(X_test, y_test))

    print("\n--- QSVM (Cirq simulator, 4 qubits) ---")
    qsvm = QSVMDetector(backend="cirq_sim", n_qubits=4)
    qsvm.fit(X_train, y_train)
    print(qsvm.report(X_test, y_test))
