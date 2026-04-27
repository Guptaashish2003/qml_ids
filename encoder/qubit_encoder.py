"""
Quantum Data Encoder
====================
Implements Section 6 of the paper: translates classical bit-represented
network stream features into qubit rotation circuits using Pauli-X (Rx) gates.

Each stream record (58 fields) is encoded as a sequence of Rx rotation gates
applied to a single qubit, where the rotation angle for field i is:

    angle_i = normalize(value_i) * π

This matches the paper's algorithm (Figure 3):
    1. IP address → rotation angle → Pauli gate
    2. For each of the 58 numeric fields: field_value → angle → Rx gate

Two backends are supported:
    - Cirq   (used in the paper, Google's framework)
    - Qiskit (IBM, alternative backend)

Usage
-----
    from encoder.qubit_encoder import QuantumEncoder
    enc = QuantumEncoder(backend="cirq")
    circuits = enc.encode_batch(X_array)          # X: (N, 58) numpy array
    tensors  = enc.to_tfq_tensors(circuits)       # for TensorFlow Quantum
"""

import numpy as np
from typing import List, Literal


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------
def _normalize_to_angle(value: float,
                         v_min: float = 0.0,
                         v_max: float = 1.0) -> float:
    """Map a scalar value to [0, π] as a rotation angle."""
    if v_max == v_min:
        return 0.0
    normed = (value - v_min) / (v_max - v_min)
    return float(np.clip(normed, 0.0, 1.0)) * np.pi


# ---------------------------------------------------------------------------
# Cirq backend
# ---------------------------------------------------------------------------
def _encode_cirq(features: np.ndarray) -> "cirq.Circuit":
    """
    Encode one stream record as a Cirq circuit (single qubit, 58 Rx gates).
    Mirrors the paper's code snippet exactly.
    """
    import cirq

    qubit   = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit()

    for val in features:
        angle = _normalize_to_angle(float(val))
        circuit.append(cirq.rx(angle)(qubit))

    return circuit


# ---------------------------------------------------------------------------
# Qiskit backend
# ---------------------------------------------------------------------------
def _encode_qiskit(features: np.ndarray) -> "QuantumCircuit":
    """
    Encode one stream record as a Qiskit QuantumCircuit (1 qubit, 58 Rx gates).
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    for val in features:
        angle = _normalize_to_angle(float(val))
        qc.rx(angle, 0)
    return qc


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------
class QuantumEncoder:
    """
    Encodes a batch of classical stream feature vectors into quantum circuits.

    Parameters
    ----------
    backend  : "cirq" | "qiskit"
    fit_scaler : bool
        If True, fit a MinMax scaler on the first call to encode_batch so that
        feature values are properly normalised to [0, 1] before angle mapping.
    """

    def __init__(
        self,
        backend: Literal["cirq", "qiskit"] = "cirq",
        fit_scaler: bool = True,
    ):
        self.backend    = backend
        self.fit_scaler = fit_scaler
        self._scaler    = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "QuantumEncoder":
        """Fit the MinMax scaler on training data."""
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(X)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            return self._scaler.transform(X)
        return np.clip(X, 0, 1)

    # ------------------------------------------------------------------
    def encode_single(self, features: np.ndarray):
        """Encode a single feature vector (1-D array of length n_features)."""
        features = self._transform(features.reshape(1, -1)).flatten()
        if self.backend == "cirq":
            return _encode_cirq(features)
        elif self.backend == "qiskit":
            return _encode_qiskit(features)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    def encode_batch(self, X: np.ndarray, verbose: bool = True) -> list:
        """
        Encode a 2-D feature matrix (N, n_features) into N quantum circuits.

        Parameters
        ----------
        X       : numpy array of shape (N, n_features)
        verbose : show progress bar

        Returns
        -------
        List of circuits (cirq.Circuit or qiskit.QuantumCircuit)
        """
        if self.fit_scaler and self._scaler is None:
            self.fit(X)

        X_scaled = self._transform(X)
        circuits  = []

        iter_range = range(len(X_scaled))
        if verbose:
            try:
                from tqdm import tqdm
                iter_range = tqdm(iter_range, desc="Encoding streams → qubits")
            except ImportError:
                pass

        for i in iter_range:
            circuits.append(self.encode_single(X_scaled[i]))

        return circuits

    # ------------------------------------------------------------------
    def to_tfq_tensors(self, circuits: list):
        """
        Convert a list of Cirq circuits to TensorFlow Quantum tensors.
        Requires: pip install tensorflow-quantum
        """
        if self.backend != "cirq":
            raise ValueError("TFQ tensors only available for the Cirq backend.")
        try:
            import tensorflow_quantum as tfq
        except ImportError:
            raise ImportError("Install tensorflow-quantum: pip install tensorflow-quantum")
        return tfq.convert_to_tensor(circuits)

    # ------------------------------------------------------------------
    def to_qiskit_statevectors(self, circuits: list) -> np.ndarray:
        """
        Simulate all Qiskit circuits and return final statevectors.
        Shape: (N, 2)  — complex amplitudes [|0⟩, |1⟩]
        """
        if self.backend != "qiskit":
            raise ValueError("Statevectors only available for the Qiskit backend.")
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit.quantum_info import Statevector

        results = []
        for qc in circuits:
            sv = Statevector.from_instruction(qc)
            results.append(sv.data)
        return np.array(results)   # shape (N, 2)

    # ------------------------------------------------------------------
    def visualize(self, features: np.ndarray, max_gates: int = 10):
        """Print a text diagram of the circuit for the first max_gates rotations."""
        single_features = features[:max_gates] if len(features) > max_gates else features
        circuit = self.encode_single(
            np.pad(single_features, (0, max(0, len(features) - len(single_features))))
            if self.backend == "cirq"
            else single_features
        )
        if self.backend == "cirq":
            print(circuit)
        else:
            print(circuit.draw(output="text"))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Quantum Encoder Demo ===\n")
    rng = np.random.default_rng(0)
    sample = rng.uniform(0, 100, size=(3, 58))

    for backend in ["cirq", "qiskit"]:
        print(f"\n--- Backend: {backend} ---")
        try:
            enc = QuantumEncoder(backend=backend)
            enc.fit(sample)
            circuits = enc.encode_batch(sample, verbose=False)
            print(f"Encoded {len(circuits)} circuits.")
            print("First circuit (first 5 gates):")
            enc.visualize(sample[0, :5])
        except ImportError as e:
            print(f"  Skipped ({e})")
