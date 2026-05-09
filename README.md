
# QML Intrusion Detection — IoT-23 Research Suite

This repository implements advanced Quantum Machine Learning (QML) techniques for network intrusion detection, specifically optimized for the IoT-23 dataset. The implementation follows the architectural principles described in:

> **"Security intrusion detection using quantum machine learning techniques"**
> *Authors: Maxim Kalinin, Vasiliy Krundyshev (2023)*
> *Journal: Journal of Computer Virology and Hacking Techniques*

## 🚀 Key Features

### 1. High-Capacity QCNN
We implemented a Quantum Convolutional Neural Network (QCNN) with **Data Re-uploading**. Unlike standard QML models, this version packs multiple network features into the same qubits through sequential rotations ($RX \to RY \to RZ$), allowing the model to capture high-dimensional traffic patterns.

### 2. Unified Multi-Class Architecture
Traditional QML models are often limited to Binary classification. We have implemented a **Unified Multi-Class Readout** using **State-Probability Mapping**:
* Maps the full $2^n$ quantum state space (16 states for 4 qubits) to the 7 traffic classes in IoT-23.
* Allows a single quantum circuit to predict all categories (DDoS, Botnet, PortScan, etc.) simultaneously.

### 3. SPSA Optimization
To solve the "Barren Plateau" and training speed issues typical in quantum simulations, we utilize **SPSA (Simultaneous Perturbation Stochastic Approximation)**:
* Reduces training time from hours to **minutes**.
* Achieves gradient estimation using only **2 simulations per batch**, regardless of the number of parameters.

### 4. Full Research Pipeline
* **Automated Data Loading**: Robust handling of large IoT-23 flow logs.
* **Classical Baseline**: Integrated 1D-CNN for direct performance comparison.
* **Evaluation Suite**: Automated generation of ROC curves, Confusion Matrices, and training time tables (Table 4 style).

---

## 🛠️ Usage

### Installation
Ensure you have the required dependencies installed:
```bash
pip install cirq tensorflow scikit-learn pandas matplotlib
```

### Running the QCNN Research Suite
To run the high-capacity multi-class QCNN against the IoT-23 dataset:
```bash
python run.qcnn.py \
  --iot23 /path/to/IoTScenarios \
  --n-samples 2000 \
  --n-qubits 4 \
  --epochs 50
```

### Running the QSVM (Binary/OvR)
To run the Quantum Support Vector Machine implementation:
```bash
python ibm.qsvm.py --iot23 /path/to/IoTScenarios --n-samples 1000
```

---

## 📊 Methodology

| Feature | Implementation |
| :--- | :--- |
| **Encoding** | ZZ-Entangled + Rotational Data Re-uploading |
| **Layers** | Quasi-local Unitaries (Conv) + Conditioned Rotations (Pool) |
| **Readout** | Z-Expectation Vector → Softmax Linear Head |
| **Optimizer** | SPSA + Adam (Robbins-Monro Scheduled) |
| **Dataset** | IoT-23 (Balanced Stratified Sampling) |

## 📈 Expected Results
The pipeline generates a results folder (`results_qcnn/`) containing:
* `combined_QCNN.png`: Side-by-side comparison of ROC and Confusion metrics.
* `roc_QCNN.png`: Per-class True Positive Rate vs False Positive Rate.
* `summary.txt`: Table 4 style training time analysis and final accuracy report.

---
*Developed as a high-performance implementation of Quantum Intrusion Detection research.*
