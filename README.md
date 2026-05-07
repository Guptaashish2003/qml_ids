# QML Intrusion Detection System
### Implementation of *"Security intrusion detection using quantum machine learning techniques"*
> Kalinin & Krundyshev (2023), Journal of Computer Virology and Hacking Techniques

---

## Project Structure

```
qml_ids/
├── data/
│   └── stream_builder.py     ← Section 5: pcap/CSV → 58-field stream dataset
├── encoder/
│   └── qubit_encoder.py      ← Section 6: classical features → qubit circuits
├── models/
│   ├── qsvm_model.py         ← Section 3.1: QSVM + classical SVM baseline
│   └── qcnn_model.py         ← Section 3.2: QCNN + classical CNN baseline
├── utils/
│   └── evaluation.py         ← Confusion matrix, ROC curves, Table 4
├── main.py                   ← End-to-end pipeline
└── requirements.txt
```

---

## Installation

```bash
# Core (required)
pip install numpy pandas scikit-learn matplotlib seaborn tqdm cirq

# Qiskit backend (optional, for IBM Q)
pip install qiskit qiskit-machine-learning qiskit-aer qiskit-algorithms

# TensorFlow Quantum backend (optional, for TFQ)
pip install tensorflow tensorflow-quantum

# Packet capture (optional, for real .pcap files)
pip install scapy
```

---

## Quick Start

### 1. Run the demo (no data needed)
```bash
python main.py --mode demo
```
Generates a synthetic 500-stream dataset and runs SVM vs QSVM comparison.

### 2. Full benchmark (Table 4 reproduction)
```bash
python main.py --mode full --n-streams 2000
```

### 3. Use your own data

**Option A — Download the IoT Network Intrusion Dataset**
```
https://ieee-dataport.org/open-access/iot-network-intrusion-dataset
```
Export with tshark:
```bash
tshark -r capture.pcap -T fields \
  -e frame.len -e frame.time_epoch -e ip.ttl \
  -e tcp.srcport -e tcp.dstport -e tcp.flags.syn \
  -e tcp.seq -e tcp.ack -e tcp.window_size_value \
  -E header=y -E separator=, > raw_packets.csv
```
Then run:
```bash
python main.py --csv raw_packets.csv
```

**Option B — Use a .pcap file directly**
```bash
python main.py --pcap capture.pcap
```

---

## Component Details

### 1. Stream Dataset Builder (`data/stream_builder.py`)
Converts packet-level data into stream-level records with **58 statistical features**:

| Feature group | Fields |
|---|---|
| TCP/IP flag averages | 13 flags (SYN, ACK, FIN, RST, …) |
| Frame length stats | mean, std, min, max, bandwidth |
| Payload stats | mean, std, min, max, printable chars |
| Port standard deviations | src, dst |
| IP TTL stats | mean, std, min, max |
| TCP seq/ack stats | mean, std, min, max × 2 |
| TCP window size stats | mean, std, min, max |
| Inter-packet interval stats | mean, std, min, max |
| Summary | count, duration, prate |

```python
from data.stream_builder import StreamBuilder
sb = StreamBuilder()
df = sb.from_csv("raw_packets.csv", label_col="label")
# or
df = StreamBuilder.generate_synthetic(n_streams=5000)
```

### 2. Quantum Encoder (`encoder/qubit_encoder.py`)
Translates each stream's 58 fields into rotation angles applied as **Rx Pauli gates** on a single qubit:

```
angle_i = normalize(value_i) × π
circuit = Rx(angle_0) · Rx(angle_1) · ... · Rx(angle_57)
```

```python
from encoder.qubit_encoder import QuantumEncoder
enc = QuantumEncoder(backend="cirq")   # or "qiskit"
enc.fit(X_train)
circuits = enc.encode_batch(X_train)
```

### 3. QSVM (`models/qsvm_model.py`)
Uses a **quantum kernel** K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|² via ZZFeatureMap (Qiskit) or Cirq statevector simulation:

```python
from models.qsvm_model import QSVMDetector, SVMBaseline

# Classical baseline
svm = SVMBaseline()
svm.fit(X_train, y_train)

# QSVM (Cirq sim — no IBM account needed)
qsvm = QSVMDetector(backend="cirq_sim", n_qubits=4)
qsvm.fit(X_train, y_train)
print(qsvm.report(X_test, y_test))
```

### 4. QCNN (`models/qcnn_model.py`)
Alternating **convolution** (parameterised 2-qubit unitaries) and **sub-sampling** (pooling via measurement) layers, trained by **parameter-shift rule**:

```python
from models.qcnn_model import QCNNDetector, CNNBaseline

qcnn = QCNNDetector(n_qubits=8, n_layers=2, backend="cirq_sim")
qcnn.fit(X_train, y_train, epochs=20)
```

---

## Research Results (IoT-23 Benchmark)

The project includes an optimized research pipeline (`run.qcnn.py`) that benchmarks Classical vs. Quantum performance on the **IoT-23** malware dataset. 

### Performance Comparison (1,000 samples, 4 Qubits)
| Metric | Classical CNN | Quantum QCNN (Research) |
|---|---|---|
| **Accuracy** | **90.50%** | **89.50%** |
| **Precision (Malicious)** | 0.90 | 0.89 |
| **Recall (Malicious)** | 0.94 | 0.94 |
| **Training Time** | ~1.6s | ~620s |

> **Key Research Achievement**: Using **Dynamic Threshold Tuning** and **Strongly Entangling Circuits**, the QCNN achieves accuracy parity with classical deep learning models on IoT-23 network flows, even with only 4 qubits.

### Reproducing Results
To reproduce the latest benchmark results:
```bash
python run.qcnn.py --iot23 /path/to/iot23 --n-samples 1000 --binary --epochs 50 --lr 0.05
```

---

## Reproducing Paper Results (Multi-Class)

---

## Hardware Notes

| Config | Qubits | Speed |
|---|---|---|
| Cirq simulator (default) | 4–8 | Fast on laptop |
| Qiskit Aer simulator | up to 30 | Requires `qiskit-aer` |
| IBM Q (real hardware) | 5–127 | Requires IBM account |

For the paper's 10M-record dataset, use `n_qubits=8` and run on a GPU-backed machine with Nvidia CUDA (as described in Section 7).

---

## Citation

```bibtex
@article{kalinin2023security,
  title={Security intrusion detection using quantum machine learning techniques},
  author={Kalinin, Maxim and Krundyshev, Vasiliy},
  journal={Journal of Computer Virology and Hacking Techniques},
  volume={19},
  pages={125--136},
  year={2023},
  publisher={Springer}
}
```
