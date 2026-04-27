# QML-Based Network Intrusion Detection System
### Implementation of *"Security Intrusion Detection Using Quantum Machine Learning Techniques"*
> Kalinin & Krundyshev (2023) — Journal of Computer Virology and Hacking Techniques

---

## Overview

This project implements a **Quantum Machine Learning (QML)** pipeline for network intrusion detection using the **IoT-23 dataset** (Stratosphere Lab / CTU). It compares classical machine learning methods (SVM, CNN) against their quantum counterparts (QSVM, QCNN) as described in the paper.

The pipeline covers the full lifecycle:
- Raw IoT network flow loading from Zeek `conn.log.labeled` files
- Data cleaning and preprocessing
- Quantum feature encoding (classical bits → qubit rotation circuits)
- Hyperparameter optimization via grid search with cross-validation
- Model training, evaluation, and visualization

---

## Experimental Results (IoT-23 Dataset)

### Model Comparison

| Model | Accuracy | Train Time | Backend |
|---|---|---|---|
| Classical SVM | 80.30% | 0.20s | scikit-learn (RBF kernel) |
| **QSVM** | **85.04%** | **3.43s** | Cirq simulator (4 qubits) |

**QSVM outperforms classical SVM by +4.74% accuracy** on the IoT-23 dataset.

---

### Per-Class Results — QSVM (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 1 |
| Benign | 0.95 | 0.35 | 0.51 | 55 |
| Botnet | 0.90 | 1.00 | **0.95** | 73 |
| C&C | 0.00 | 0.00 | 0.00 | 4 |
| DDoS | 0.73 | 0.77 | 0.75 | 39 |
| PortScan | 0.84 | 0.95 | **0.90** | 229 |
| **Overall** | **0.85** | **0.85** | **0.83** | **401** |

### Per-Class Results — Classical SVM (Baseline)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Attack | 1.00 | 1.00 | 1.00 | 1 |
| Benign | 0.95 | 0.36 | 0.53 | 55 |
| Botnet | 0.66 | 1.00 | 0.79 | 73 |
| C&C | 0.00 | 0.00 | 0.00 | 4 |
| DDoS | 1.00 | 0.77 | 0.87 | 39 |
| PortScan | 0.83 | 0.86 | 0.85 | 229 |
| **Overall** | **0.83** | **0.80** | **0.79** | **401** |

---

### Dataset Statistics

| Property | Value |
|---|---|
| Source | Stratosphere Lab — IoT-23 (CTU) |
| Captures loaded | 20 CTU-IoT-Malware-Capture folders |
| Total flows | 1,444,706 |
| Features after preprocessing | 20 |
| Training samples | 1,600 |
| Test samples | 401 |

**Class distribution in sample (2,001 flows):**

| Class | Count | % |
|---|---|---|
| PortScan | 1,144 | 57.2% |
| Botnet | 364 | 18.2% |
| DDoS | 192 | 9.6% |
| Benign | 274 | 13.7% |
| C&C | 22 | 1.1% |
| Attack | 5 | 0.2% |

---

### Hyperparameter Optimization Results

**SVM Grid Search (3-fold CV):**
| C | Kernel | CV Accuracy |
|---|---|---|
| 0.1 | RBF | 81.31% |
| **1.0** | **RBF** | **82.00% ← best** |

**QSVM Grid Search (3-fold CV, Cirq simulator):**
| n_qubits | C | CV Accuracy |
|---|---|---|
| 4 | 0.1 | 77.12% |
| **4** | **1.0** | **87.31% ← best** |

---

## Project Structure

```
qml_ids/
├── data/
│   ├── iot23_loader.py       ← IoT-23 conn.log.labeled reader & preprocessor
│   ├── stream_builder.py     ← Stream dataset builder (tshark CSV / pcap)
│   └── nslkdd_loader.py      ← NSL-KDD dataset loader (optional)
├── encoder/
│   └── qubit_encoder.py      ← Classical features → Rx qubit rotation circuits
├── models/
│   ├── qsvm_model.py         ← QSVM (quantum kernel) + SVM baseline
│   └── qcnn_model.py         ← QCNN + CNN baseline
├── utils/
│   └── evaluation.py         ← Confusion matrix, ROC curves, Table 4
├── results/
│   ├── combined_QSVM_IoT23.png   ← ROC + confusion matrix (QSVM)
│   ├── combined_SVM_IoT23.png    ← ROC + confusion matrix (SVM)
│   ├── qsvm_model.joblib         ← Saved QSVM model
│   └── svm_model.joblib          ← Saved SVM model
├── main.py                   ← Full pipeline entry point
├── QML_IoT23.ipynb           ← Jupyter notebook (IoT-23)
├── QML_NSL_KDD.ipynb         ← Jupyter notebook (NSL-KDD)
└── requirements.txt
```

---

## Installation

```bash
# Create virtual environment
python -m venv qml-env
source qml-env/bin/activate        # Linux/Mac
# qml-env\Scripts\activate         # Windows

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn cirq tqdm joblib
```

---

## Dataset Setup

This project uses the **IoT-23 dataset** from Stratosphere Lab (CTU).

**Your dataset directory structure:**
```
IoTScenarios/
├── CTU-IoT-Malware-Capture-1-1/
│   └── bro/
│       └── conn.log.labeled     ← Zeek flow log with labels
├── CTU-IoT-Malware-Capture-3-1/
│   └── bro/
│       └── conn.log.labeled
└── ... (20 captures total)
```

**File format:** Tab-separated Zeek conn.log with space-separated label fields at the end:
```
#separator \x09
#fields  ts  uid  id.orig_h  id.orig_p  ... tunnel_parents  label  detailed-label
1525879831.01  CUmr...  192.168.100.103  51524  ...  (empty)   Malicious   PartOfAHorizontalPortScan
```

---

## Running the Pipeline

```bash
cd qml_ids

# Standard run (2000 samples, with optimizer):
python main.py --iot23 /path/to/IoTScenarios --n-samples 2000

# Skip optimizer (faster, uses default params):
python main.py --iot23 /path/to/IoTScenarios --skip-optimize --n-samples 500

# Binary classification (Benign vs Malicious):
python main.py --iot23 /path/to/IoTScenarios --binary

# Full pipeline with QCNN + Table 4 benchmark:
python main.py --iot23 /path/to/IoTScenarios --mode full

# Synthetic demo (no dataset needed):
python main.py --demo
```

---

## Pipeline Steps

### Step 1 — Load
Reads all 20 `conn.log.labeled` files. Handles the mixed tab+space separator format specific to IoT-23. Labels are mapped from raw Zeek values:

| Raw Label | Mapped To |
|---|---|
| `Benign` / `-` | Benign |
| `PartOfAHorizontalPortScan` | PortScan |
| `DDoS` | DDoS |
| `C&C` | C&C |
| `Okiru` / `Mirai` / `Torii` | Botnet |
| others | Malicious / Attack |

### Step 2 — Clean
- Remove NaN / Inf rows
- Remove zero-variance features (6 removed in this run)
- Clip outliers at mean ± 5σ (winsorization)

### Step 3 — Quantum Encoding
Each flow's features are encoded as **Rx rotation gates** on a single qubit:

```
angle_i = normalize(feature_i) × π
circuit = Rx(angle_0) · Rx(angle_1) · ... · Rx(angle_19)
```

Sample circuit (6 of 20 gates):
```
(1, 1): ───Rx(0)───Rx(0)───Rx(0)───Rx(0.333π)───Rx(0.222π)───Rx(0)───
```

### Step 4 — Hyperparameter Optimizer
3-fold stratified CV grid search over:
- SVM: `kernel ∈ {rbf, linear}`, `C ∈ {0.1, 1.0, 10.0}`
- QSVM: `n_qubits ∈ {2, 4}`, `C ∈ {0.1, 1.0, 10.0}`

### Step 5 — Train & Evaluate
Trains SVM and QSVM with best params, generates:
- Classification report (precision, recall, F1 per class)
- Normalised confusion matrix
- One-vs-rest ROC curves per class
- Saves plots to `results/`

### Step 6 — Summary
Prints final accuracy and training time comparison table.

---

## Quantum Architecture

### QSVM — Quantum Kernel
The quantum kernel is computed as:

```
K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
```

Where `|φ(x)⟩` is the angle-encoded quantum state. Features are first reduced to `n_qubits=4` dimensions via PCA, then encoded as Rx rotations. The kernel matrix replaces the classical RBF kernel in an SVC.

### QCNN — Quantum Convolutional Network
Alternating layers of:
- **Convolution**: parameterised 2-qubit unitaries (Rx, Rz, CNOT) in brick-wall pattern
- **Pooling**: measure source qubits, apply conditioned Rz to sink qubits
- **Fully connected**: final unitary F on remaining qubits
- **Output**: Z-expectation on readout qubit → class score

Trained via the **parameter-shift rule** (exact quantum gradient).

---

## Key Observations

1. **QSVM detects Botnet with 100% recall** (vs 100% for SVM too) — both models excel here
2. **QSVM improves PortScan recall to 95%** (vs 86% for SVM) — significant improvement
3. **DDoS recall is 77% for both** — class confusion with Botnet (21% of DDoS predicted as Botnet)
4. **C&C recall is 0% for both** — only 4 test samples, severe class imbalance
5. **Benign recall is low (35%)** — many benign flows misclassified as PortScan due to dataset imbalance (57% PortScan)
6. **QSVM ROC AUC** — Botnet: 1.00, Attack: 1.00, DDoS: 0.97, PortScan: 0.93

---

## Limitations & Future Work

- **Class imbalance**: PortScan dominates (57%) causing poor recall on minority classes (C&C, Attack). Apply SMOTE or class weighting.
- **Simulator speed**: Cirq simulator is slower than real quantum hardware. Use IBM Q for production.
- **n_qubits=4**: Only 4 features used after PCA reduction. More qubits = richer quantum kernel but slower simulation.
- **QCNN**: Not yet benchmarked on IoT-23 — expected to outperform QSVM per paper (ROC curves show QCNN > QSVM).

---

## Citation

```bibtex
@article{kalinin2023security,
  title   = {Security intrusion detection using quantum machine learning techniques},
  author  = {Kalinin, Maxim and Krundyshev, Vasiliy},
  journal = {Journal of Computer Virology and Hacking Techniques},
  volume  = {19},
  pages   = {125--136},
  year    = {2023},
  doi     = {10.1007/s11416-022-00435-0}
}
```

**Dataset:**
```
Garcia, S., Parmisano, A., Erquiaga, M.J.
IoT-23: A labeled dataset with malicious and benign IoT network traffic (Version 1.0.0)
Stratosphere Laboratory, Czech Technical University in Prague, 2020
https://www.stratosphereips.org/datasets-iot23
```