"""
NSL-KDD Dataset Loader
=======================
Loads, preprocesses, and prepares the NSL-KDD dataset for the QML pipeline.

NSL-KDD has 41 features:
  - 38 numeric features  (used directly)
  - 3 categorical features: protocol_type, service, flag (one-hot encoded)

Attack categories (5 classes used in this implementation):
  - Normal
  - DoS      (Denial of Service)
  - Probe    (Surveillance / scanning)
  - R2L      (Remote to Local)
  - U2R      (User to Root)

Download the dataset from:
  https://www.unb.ca/cic/datasets/nsl.html
  Files needed: KDDTrain+.txt  and  KDDTest+.txt

Or just call NSLKDDLoader.download() and it will fetch them automatically.

Usage
-----
    from data.nslkdd_loader import NSLKDDLoader

    loader = NSLKDDLoader()
    loader.download()                          # auto-download if needed
    X_train, y_train = loader.load_train()
    X_test,  y_test  = loader.load_test()
    print(loader.feature_names)               # list of final feature names
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Column names (41 features + label + difficulty)
# ---------------------------------------------------------------------------
COL_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label", "difficulty_level",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
NUMERIC_COLS = [
    c for c in COL_NAMES
    if c not in CATEGORICAL_COLS + ["label", "difficulty_level"]
]

# ---------------------------------------------------------------------------
# Attack → category mapping (5-class)
# ---------------------------------------------------------------------------
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm",
}
PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint",
}
R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "httptunnel",
}
U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm",
}


def _map_label(label: str) -> str:
    label = label.lower().strip()
    if label == "normal":
        return "Normal"
    if label in DOS_ATTACKS:
        return "DoS"
    if label in PROBE_ATTACKS:
        return "Probe"
    if label in R2L_ATTACKS:
        return "R2L"
    if label in U2R_ATTACKS:
        return "U2R"
    # Unknown attacks → classify by closest known category
    return "DoS"   # safe fallback for unseen attack types in test set


# ---------------------------------------------------------------------------
# NSLKDDLoader
# ---------------------------------------------------------------------------
class NSLKDDLoader:
    """
    Loads and preprocesses the NSL-KDD dataset.

    Parameters
    ----------
    data_dir   : directory where KDDTrain+.txt / KDDTest+.txt are stored
    binary     : if True, use binary labels (Normal / Attack) instead of 5-class
    """

    TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

    def __init__(self, data_dir: str = "data/nslkdd", binary: bool = False):
        self.data_dir = Path(data_dir)
        self.binary   = binary
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names: List[str] = []
        self._ohe_cols: List[str]     = []

    # ------------------------------------------------------------------
    def download(self, force: bool = False):
        """Download KDDTrain+.txt and KDDTest+.txt if not already present."""
        import urllib.request

        files = {
            "KDDTrain+.txt": self.TRAIN_URL,
            "KDDTest+.txt" : self.TEST_URL,
        }
        for fname, url in files.items():
            dest = self.data_dir / fname
            if dest.exists() and not force:
                print(f"[NSLKDDLoader] Found {dest} — skipping download.")
                continue
            print(f"[NSLKDDLoader] Downloading {fname} …")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"[NSLKDDLoader] Saved → {dest}")
            except Exception as e:
                print(f"[NSLKDDLoader] Download failed: {e}")
                print(f"  Please download manually from:")
                print(f"  https://www.unb.ca/cic/datasets/nsl.html")
                print(f"  and place KDDTrain+.txt / KDDTest+.txt in: {self.data_dir}/")

    # ------------------------------------------------------------------
    def _read_raw(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=COL_NAMES)
        print(f"[NSLKDDLoader] Loaded {len(df):,} records from {path.name}")
        return df

    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline:
          1. Drop difficulty_level column
          2. Map labels → 5-class (or binary)
          3. One-hot encode categorical columns
          4. MinMax scale all numeric features to [0, π]  (ready for Rx gates)
          5. Return X (numpy), y (string labels)
        """
        df = df.copy()

        # Step 1 – drop difficulty
        df.drop(columns=["difficulty_level"], inplace=True, errors="ignore")

        # Step 2 – label mapping
        y = df["label"].apply(_map_label if not self.binary
                              else lambda l: "Normal" if l.lower() == "normal" else "Attack")
        df.drop(columns=["label"], inplace=True)

        # Step 3 – one-hot encode categorical columns
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS)

        if fit:
            self._ohe_cols      = list(df.columns)
            self.feature_names  = self._ohe_cols
        else:
            # Align test columns with training columns (add missing, drop extra)
            for col in self._ohe_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[self._ohe_cols]

        X_raw = df.values.astype(np.float32)

        # Step 4 – MinMax scale to [0, π]
        if fit:
            from sklearn.preprocessing import MinMaxScaler
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
            X = self._scaler.fit_transform(X_raw)
        else:
            X = self._scaler.transform(X_raw)

        print(f"[NSLKDDLoader] Preprocessed: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"               Classes: { {c: int((y == c).sum()) for c in sorted(y.unique())} }")
        return X, y.values

    # ------------------------------------------------------------------
    def load_train(
        self,
        path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the training set."""
        p = Path(path) if path else self.data_dir / "KDDTrain+.txt"
        if not p.exists():
            print(f"[NSLKDDLoader] {p} not found. Calling download() …")
            self.download()
        df = self._read_raw(p)
        return self._preprocess(df, fit=True)

    # ------------------------------------------------------------------
    def load_test(
        self,
        path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the test set (must call load_train first)."""
        p = Path(path) if path else self.data_dir / "KDDTest+.txt"
        if not p.exists():
            print(f"[NSLKDDLoader] {p} not found. Calling download() …")
            self.download()
        df = self._read_raw(p)
        return self._preprocess(df, fit=False)

    # ------------------------------------------------------------------
    def load_sample(
        self,
        n: int = 1000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a stratified sample from the training set.
        Useful for quick experiments on the QML simulator.
        """
        X, y = self.load_train()
        from sklearn.model_selection import train_test_split
        if n >= len(X):
            return X, y
        _, X_s, _, y_s = train_test_split(
            X, y, test_size=n / len(X),
            stratify=y, random_state=random_state)
        print(f"[NSLKDDLoader] Sampled {len(X_s):,} records.")
        return X_s, y_s

    # ------------------------------------------------------------------
    @staticmethod
    def describe():
        """Print dataset summary."""
        print("""
NSL-KDD Dataset Summary
------------------------
Source : Canadian Institute for Cybersecurity (UNB)
URL    : https://www.unb.ca/cic/datasets/nsl.html

Files  : KDDTrain+.txt  (~125,973 records)
         KDDTest+.txt   (~ 22,544 records)

Features (41 total):
  • 38 numeric  — duration, src_bytes, dst_bytes, rates, counts …
  • 3 categorical — protocol_type (tcp/udp/icmp)
                    service (http, ftp, smtp, …)
                    flag (SF, S0, REJ, …)

After one-hot encoding: ~122 features

Attack classes (5-class mode):
  Normal  — benign traffic
  DoS     — Denial of Service (neptune, smurf, back, …)
  Probe   — Port/network scanning (ipsweep, nmap, …)
  R2L     — Remote to Local exploits (ftp_write, guess_passwd, …)
  U2R     — Privilege escalation (buffer_overflow, rootkit, …)
""")
