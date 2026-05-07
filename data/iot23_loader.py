"""
IoT-23 Dataset Loader  (Stratosphere Lab / CTU)
================================================
Reads the `conn.log.labeled` Zeek files from each CTU-IoT-Malware-Capture-*
folder and converts them into a feature matrix ready for the QML pipeline.

Dataset structure on disk
--------------------------
  IoT-23/
  ├── CTU-IoT-Malware-Capture-1-1/
  │   └── bro/
  │       └── conn.log.labeled          ← main file we read
  ├── CTU-IoT-Malware-Capture-3-1/
  │   └── conn.log.labeled              ← sometimes at root
  └── ...

Zeek conn.log.labeled columns (tab-separated)
----------------------------------------------
  ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p,
  proto, service, duration, orig_bytes, resp_bytes,
  conn_state, local_orig, local_resp, missed_bytes,
  history, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes,
  tunnel_parents, label, detailed-label

Label mapping used
------------------
  Benign              → "Benign"
  C&C                 → "C&C"
  DDoS                → "DDoS"
  PartOfAHorizontalPortScan → "PortScan"
  Attack              → "Attack"
  FileDownload        → "FileDownload"
  (anything else)     → "Malicious"

Usage
-----
    from data.iot23_loader import IoT23Loader

    loader = IoT23Loader(dataset_dir="path/to/IoT-23-folders")
    X, y   = loader.load_all()          # load all captures
    X, y   = loader.load_capture("CTU-IoT-Malware-Capture-1-1")
    X, y   = loader.load_sample(n=2000) # stratified sample
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Zeek conn.log column names
# ---------------------------------------------------------------------------
CONN_LOG_COLS = [
    "ts", "uid",
    "id.orig_h", "id.orig_p",
    "id.resp_h", "id.resp_p",
    "proto", "service", "duration",
    "orig_bytes", "resp_bytes", "conn_state",
    "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]

# Features we extract as numeric columns
NUMERIC_FEATURES = [
    "duration", "orig_bytes", "resp_bytes",
    "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes",
]

# Categorical features we one-hot encode
CATEGORICAL_FEATURES = ["proto", "conn_state"]


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------
def _normalise_label(label: str) -> str:
    """Map Zeek detailed label to a clean category."""
    if pd.isna(label):
        return "Benign"
    label = str(label).strip()
    if label in ("-", "Benign", "(empty)"):
        return "Benign"
    if "PortScan" in label or "HorizontalPortScan" in label:
        return "PortScan"
    if "DDoS" in label:
        return "DDoS"
    if "C&C" in label:
        return "C&C"
    if "Attack" in label:
        return "Attack"
    if "FileDownload" in label or "HeartBeat" in label:
        return "FileDownload"
    if "Okiru" in label or "Mirai" in label or "Torii" in label:
        return "Botnet"
    return "Malicious"


# ---------------------------------------------------------------------------
# IoT23Loader
# ---------------------------------------------------------------------------
class IoT23Loader:
    """
    Loads the IoT-23 (CTU-IoT-Malware-Capture) dataset.

    Parameters
    ----------
    dataset_dir : root directory containing CTU-IoT-Malware-Capture-* folders
    binary      : if True, collapse all malicious labels into "Malicious"
    max_rows_per_capture : cap rows per capture to avoid memory issues
                          (None = load all)
    """

    def __init__(
        self,
        dataset_dir: str = ".",
        binary: bool = False,
        max_rows_per_capture: Optional[int] = 100_000,
    ):
        self.dataset_dir          = Path(dataset_dir)
        self.binary               = binary
        self.max_rows_per_capture = max_rows_per_capture
        self._scaler              = None
        self._ohe_cols: List[str] = []
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    def _find_conn_logs(self, capture_dir: Path) -> List[Path]:
        """Find all conn.log.labeled files inside a capture folder."""
        candidates = list(capture_dir.rglob("conn.log.labeled"))
        if not candidates:
            # Some captures use just 'conn.log'
            candidates = list(capture_dir.rglob("conn.log"))
        return candidates

    # ------------------------------------------------------------------
    def _read_conn_log(self, path: Path) -> pd.DataFrame:
        """
        Read a Zeek conn.log.labeled file.

        Exact format (from your files):
          #separator \\x09          ← TAB separated
          #set_separator  ,
          #empty_field    (empty)
          #unset_field    -         ← missing values are '-'
          #fields ts uid id.orig_h id.orig_p ... label detailed-label
          #types  time string addr  port     ...
          <data rows>
        """
        header      = None
        empty_field = "(empty)"
        unset_field = "-"
        rows        = []

        with open(path, "r", errors="replace") as f:
            for line in f:
                if self.max_rows_per_capture is not None and len(rows) >= self.max_rows_per_capture:
                    break
                line = line.rstrip("\n\r")

                # Parse meta-directives
                if line.startswith("#separator"):
                    # e.g. "#separator \x09"  → already using \t split, no action needed
                    continue
                if line.startswith("#empty_field"):
                    empty_field = line.split("\t")[-1].strip()
                    continue
                if line.startswith("#unset_field"):
                    unset_field = line.split("\t")[-1].strip()
                    continue
                if line.startswith("#fields"):
                    # "#fields\tts\tuid\t..."  — first token is "#fields", rest are column names
                    # The last tab-field may contain space-separated names
                    # (e.g. "tunnel_parents   label   detailed-label")
                    raw = line.split("\t")[1:]
                    last = raw[-1].strip().split()
                    header = raw[:-1] + last
                    continue
                if line.startswith("#"):
                    # Skip all other comment/meta lines (#types, #path, #open, #close)
                    continue
                if not line.strip():
                    continue

                # Split on tab first, then handle space-separated label fields
                # Split on tab first
                parts = line.split("\t")
                # The last element contains space-separated label fields
                # e.g. "(empty)   Malicious   PartOfAHorizontalPortScan"
                last_field = parts[-1].strip()
                sub_parts  = last_field.split()   # splits on any whitespace
                # sub_parts = ["(empty)", "Malicious", "PartOfAHorizontalPortScan"]
                # OR        = ["Benign", "-"]
                # Replace last tab-field with the expanded sub-parts
                # tunnel_parents is field 21 (index 20), it may be "(empty)"
                # fields 22 and 23 are label and detailed-label, space-separated in last tab field
                # So we only expand the LAST field if it's not already a label field
                if len(parts) <= 21:
                    parts = parts[:-1] + sub_parts
                rows.append(parts)

        if not rows:
            print(f"  [warn] No data rows found in {path}")
            return pd.DataFrame()

        # Use header from #fields if found, else fall back to known columns
        if header is None:
            header = CONN_LOG_COLS
            print(f"  [warn] No #fields line found in {path}, using default columns")

        n = len(header)
        # Pad short rows, trim long rows
        rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
        df   = pd.DataFrame(rows, columns=header)

        # Replace Zeek sentinel values with NaN
        df.replace(unset_field,  np.nan, inplace=True)
        df.replace(empty_field,  np.nan, inplace=True)
        df.replace("",           np.nan, inplace=True)

        return df

    # ------------------------------------------------------------------
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw Zeek conn.log into numeric feature rows.
        """
        if df.empty:
            return pd.DataFrame()

        # Normalise column names (Zeek uses id.orig_h style)
        df.columns = [c.strip() for c in df.columns]

        # Pick label column (could be 'label' or 'detailed-label')
        label_col = None
        for candidate in ["detailed-label", "label"]:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            df["label_clean"] = "Unknown"
        else:
            df["label_clean"] = df[label_col].apply(_normalise_label)

        if self.binary:
            df["label_clean"] = df["label_clean"].apply(
                lambda x: "Benign" if x == "Benign" else "Malicious"
            )

        # Convert numeric columns (NaN already replaced in _read_conn_log)
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode port numbers as numeric features too
        for port_col in ["id.orig_p", "id.resp_p"]:
            if port_col in df.columns:
                df[port_col] = pd.to_numeric(df[port_col], errors="coerce")
                # Add to numeric features if not already there
                if port_col not in NUMERIC_FEATURES:
                    NUMERIC_FEATURES.append(port_col)

        # Compute derived features
        orig_b = df.get("orig_bytes", pd.Series(0, index=df.index)).fillna(0)
        resp_b = df.get("resp_bytes", pd.Series(0, index=df.index)).fillna(0)
        orig_p = df.get("orig_pkts", pd.Series(0, index=df.index)).fillna(0)
        resp_p = df.get("resp_pkts", pd.Series(0, index=df.index)).fillna(0)
        dur    = df.get("duration", pd.Series(0, index=df.index)).fillna(0)

        df["bytes_ratio"] = orig_b / (resp_b + 1e-9)
        df["pkt_ratio"]   = orig_p / (resp_p + 1e-9)
        df["bytes_per_pkt"] = (orig_b + resp_b) / (orig_p + resp_p + 1e-9)

        # Additional derived features for better accuracy
        df["total_bytes"]    = orig_b + resp_b
        df["total_pkts"]     = orig_p + resp_p
        df["byte_asymmetry"] = (orig_b - resp_b) / (orig_b + resp_b + 1e-9)
        df["pkt_asymmetry"]  = (orig_p - resp_p) / (orig_p + resp_p + 1e-9)
        df["bytes_per_sec"]  = (orig_b + resp_b) / (dur + 1e-9)
        df["pkts_per_sec"]   = (orig_p + resp_p) / (dur + 1e-9)
        df["avg_pkt_size_orig"] = orig_b / (orig_p + 1e-9)
        df["avg_pkt_size_resp"] = resp_b / (resp_p + 1e-9)

        return df

    # ------------------------------------------------------------------
    def _to_matrix(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Convert processed DataFrame to (X, y) arrays."""
        if df.empty:
            return np.array([]), np.array([])

        y = df["label_clean"].values

        # Select feature columns
        derived = [
            "bytes_ratio", "pkt_ratio", "bytes_per_pkt",
            "total_bytes", "total_pkts",
            "byte_asymmetry", "pkt_asymmetry",
            "bytes_per_sec", "pkts_per_sec",
            "avg_pkt_size_orig", "avg_pkt_size_resp",
        ]
        numeric_cols = NUMERIC_FEATURES + derived
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        # One-hot encode categoricals
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            cat_df = pd.get_dummies(df[cat_cols], columns=cat_cols)
        else:
            cat_df = pd.DataFrame(index=df.index)

        X_num = df[numeric_cols].fillna(0).values.astype(np.float32)
        X_cat = cat_df.values.astype(np.float32)
        X_raw = np.hstack([X_num, X_cat]) if X_cat.shape[1] > 0 else X_num

        if fit:
            self._ohe_cols     = numeric_cols + list(cat_df.columns)
            self.feature_names = self._ohe_cols

            from sklearn.preprocessing import MinMaxScaler
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
            X = self._scaler.fit_transform(X_raw)
        else:
            # Align columns
            cat_df_aligned = cat_df.reindex(
                columns=[c for c in self._ohe_cols if c not in numeric_cols],
                fill_value=0
            )
            X_cat = cat_df_aligned.values.astype(np.float32)
            X_raw = np.hstack([X_num, X_cat]) if X_cat.shape[1] > 0 else X_num

            # Pad/trim to match scaler's expected input
            n_expected = len(self._scaler.scale_)
            if X_raw.shape[1] < n_expected:
                pad = np.zeros((len(X_raw), n_expected - X_raw.shape[1]), dtype=np.float32)
                X_raw = np.hstack([X_raw, pad])
            elif X_raw.shape[1] > n_expected:
                X_raw = X_raw[:, :n_expected]

            X = self._scaler.transform(X_raw)

        return X, y

    # ------------------------------------------------------------------
    def list_captures(self) -> List[str]:
        """List all CTU-IoT-Malware-Capture-* folders found."""
        captures = sorted([
            d.name for d in self.dataset_dir.iterdir()
            if d.is_dir() and "CTU-IoT" in d.name
        ])
        return captures

    # ------------------------------------------------------------------
    def load_capture(
        self,
        capture_name: str,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single capture folder."""
        capture_dir = self.dataset_dir / capture_name
        logs = self._find_conn_logs(capture_dir)

        if not logs:
            raise FileNotFoundError(
                f"No conn.log.labeled found in {capture_dir}\n"
                f"Expected file: {capture_dir}/conn.log.labeled\n"
                f"or:            {capture_dir}/bro/conn.log.labeled"
            )

        dfs = []
        for log_path in logs:
            print(f"[IoT23Loader] Reading {log_path.relative_to(self.dataset_dir)} …")
            df = self._read_conn_log(log_path)
            df = self._extract_features(df)
            if self.max_rows_per_capture is not None:
                df = df.head(self.max_rows_per_capture)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[IoT23Loader] {capture_name}: {len(combined):,} flows")
        return self._to_matrix(combined, fit=fit_scaler)

    # ------------------------------------------------------------------
    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and concatenate all CTU-IoT-Malware-Capture folders.
        First capture fits the scaler; rest transform with it.
        """
        captures = self.list_captures()
        if not captures:
            raise FileNotFoundError(
                f"No CTU-IoT-Malware-Capture-* folders found in: {self.dataset_dir}\n"
                "Make sure dataset_dir points to the folder that CONTAINS "
                "the CTU-IoT-Malware-Capture-* subfolders."
            )

        print(f"[IoT23Loader] Found {len(captures)} capture(s): {captures}")
        all_X, all_y = [], []

        for i, cap in enumerate(captures):
            try:
                X, y = self.load_capture(cap, fit_scaler=(i == 0))
                if len(X):
                    all_X.append(X)
                    all_y.append(y)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue

        if not all_X:
            raise RuntimeError("No data loaded — check your dataset_dir path.")

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)

        print(f"\n[IoT23Loader] Total: {len(X_all):,} flows × {X_all.shape[1]} features")
        classes, counts = np.unique(y_all, return_counts=True)
        for c, n in zip(classes, counts):
            print(f"   {c:30s}: {n:>8,}")

        return X_all, y_all

    # ------------------------------------------------------------------
    def load_sample(
        self,
        n: int = 2000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all captures then return a stratified sample of size n."""
        X, y = self.load_all()
        from sklearn.model_selection import train_test_split
        if n >= len(X):
            return X, y
        _, X_s, _, y_s = train_test_split(
            X, y, test_size=n / len(X),
            stratify=y, random_state=random_state
        )
        print(f"[IoT23Loader] Stratified sample: {len(X_s):,} flows")
        return X_s, y_s


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Path to folder containing CTU-IoT-Malware-Capture-* dirs")
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()

    loader = IoT23Loader(dataset_dir=args.dir)
    print("Captures found:", loader.list_captures())

    X, y = loader.load_sample(n=args.sample)
    print(f"\nShape: {X.shape}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
