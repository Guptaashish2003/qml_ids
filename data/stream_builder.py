"""
Stream Dataset Builder
======================
Implements Section 5 of the paper: converts raw network packets (pcap or
IEEEDataPort IoT Network Intrusion Dataset CSV) into stream-level records
with 58 statistical features per stream.

Supports two input modes:
  1. CSV  – pre-exported from Wireshark / tshark (column names as in the dataset)
  2. PCAP – live capture or .pcap file parsed with Scapy

Usage
-----
    from data.stream_builder import StreamBuilder
    sb = StreamBuilder()
    df = sb.from_csv("raw_packets.csv", label_col="label")
    df.to_csv("streams.csv", index=False)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: map an IP string to a float in [0, 1]
# ---------------------------------------------------------------------------
def ip_to_float(ip: str) -> float:
    try:
        parts = [int(p) for p in str(ip).split(".")]
        val = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
        return val / 0xFFFFFFFF
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Core aggregation logic (packet-level → stream-level)
# ---------------------------------------------------------------------------
def _agg_stats(series: pd.Series) -> dict:
    """Return mean, std, min, max for a numeric series."""
    return {
        "mean": series.mean(),
        "std": series.std(ddof=0),
        "min": series.min(),
        "max": series.max(),
    }


def _build_stream_record(grp: pd.DataFrame) -> dict:
    """
    Convert a group of packets (one stream) into a single feature record.
    Column names follow the IoT Network Intrusion Dataset conventions.
    Missing columns fall back to 0.
    """

    def col(name, default=0.0):
        return grp[name] if name in grp.columns else pd.Series([default] * len(grp))

    rec = {}

    # --- TCP/IP flag averages (13 flags) ---
    for flag in [
        "ip.flags.rb", "ip.flags.df", "ip.flags.mf",
        "tcp.flags.res", "tcp.flags.ns", "tcp.flags.cwr",
        "tcp.flags.ecn", "tcp.flags.urg", "tcp.flags.ack",
        "tcp.flags.push", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.fin",
    ]:
        rec[f"{flag}.mean"] = col(flag).mean()

    # --- Frame length stats ---
    fl = col("frame.len")
    for k, v in _agg_stats(fl).items():
        rec[f"frame.len.{k}"] = v
    rec["frame.len.rate"] = fl.sum() / max(grp["duration"].iloc[0] if "duration" in grp.columns else 1.0, 1e-9)

    # --- Payload stats ---
    pay = col("payload")
    for k, v in _agg_stats(pay).items():
        rec[f"payload.{k}.mean"] = v
    rec["payload.print.mean"] = col("payload.print").mean()

    # --- Port standard deviations ---
    rec["srcport.std"] = col("srcport").std(ddof=0)
    rec["dstport.std"] = col("dstport").std(ddof=0)

    # --- IP TTL stats ---
    ttl = col("ip.ttl")
    for k, v in _agg_stats(ttl).items():
        rec[f"ip.ttl.{k}"] = v

    # --- TCP sequence number stats ---
    seq = col("tcp.seq_raw")
    for k, v in _agg_stats(seq).items():
        rec[f"tcp.seq_raw.{k}"] = v

    # --- TCP ACK number stats ---
    ack = col("tcp.ack_raw")
    for k, v in _agg_stats(ack).items():
        rec[f"tcp.ack_raw.{k}"] = v

    # --- TCP window size stats ---
    win = col("tcp.window_size_value")
    for k, v in _agg_stats(win).items():
        rec[f"tcp.window_size_value.{k}"] = v

    # --- Inter-packet interval stats ---
    if "frame.time_epoch" in grp.columns:
        times = grp["frame.time_epoch"].sort_values()
        intervals = times.diff().dropna()
    else:
        intervals = pd.Series([0.0])
    for k, v in _agg_stats(intervals).items():
        rec[f"int.{k}"] = v

    # --- Summary fields ---
    rec["count"]    = len(grp)
    rec["duration"] = grp["duration"].iloc[0] if "duration" in grp.columns else 0.0
    rec["prate"]    = rec["count"] / max(rec["duration"], 1e-9)

    return rec


# ---------------------------------------------------------------------------
# StreamBuilder class
# ---------------------------------------------------------------------------
class StreamBuilder:
    """
    Builds the 58-field stream dataset from raw packet data.
    """

    def __init__(self, stream_window: int = 100):
        """
        Parameters
        ----------
        stream_window : int
            Number of packets to group into one stream when no stream-id
            column is available.
        """
        self.stream_window = stream_window

    # ------------------------------------------------------------------
    def from_csv(
        self,
        csv_path: str,
        label_col: Optional[str] = "label",
        stream_col: Optional[str] = "stream",
        sep: str = ",",
    ) -> pd.DataFrame:
        """
        Build stream dataset from a tshark-exported CSV.

        Parameters
        ----------
        csv_path   : path to the CSV file
        label_col  : column containing the traffic label (attack type / normal)
        stream_col : column used for grouping packets into streams;
                     if absent, packets are grouped by sliding window
        sep        : CSV delimiter
        """
        print(f"[StreamBuilder] Reading {csv_path} …")
        df = pd.read_csv(csv_path, sep=sep, low_memory=False)
        print(f"[StreamBuilder] Loaded {len(df):,} packets, {len(df.columns)} columns.")

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Merge TCP/UDP port columns → unified srcport / dstport
        if "tcp.srcport" in df.columns and "srcport" not in df.columns:
            df["srcport"] = df.get("tcp.srcport", 0).fillna(df.get("udp.srcport", 0))
            df["dstport"] = df.get("tcp.dstport", 0).fillna(df.get("udp.dstport", 0))

        # Merge tcp.stream / udp.stream → stream
        if stream_col not in df.columns:
            if "tcp.stream" in df.columns:
                df[stream_col] = df["tcp.stream"].fillna(df.get("udp.stream", np.nan))
            else:
                # Fall back to sliding-window grouping
                df[stream_col] = df.index // self.stream_window

        # Compute per-packet duration proxy (0 for individual packets)
        if "duration" not in df.columns:
            if "frame.time_epoch" in df.columns:
                t = df.groupby(stream_col)["frame.time_epoch"]
                df["duration"] = df[stream_col].map(t.transform(lambda s: s.max() - s.min()))
            else:
                df["duration"] = 0.0

        print(f"[StreamBuilder] Grouping into streams …")
        records = []
        groups  = df.groupby(stream_col)
        for _, grp in groups:
            rec = _build_stream_record(grp)
            if label_col and label_col in grp.columns:
                rec["label"] = grp[label_col].mode()[0]
            records.append(rec)

        result = pd.DataFrame(records).fillna(0.0)
        print(f"[StreamBuilder] Produced {len(result):,} stream records, {len(result.columns)} fields.")
        return result

    # ------------------------------------------------------------------
    def from_pcap(self, pcap_path: str, label: str = "unknown") -> pd.DataFrame:
        """
        Build stream dataset directly from a .pcap file using Scapy.
        Requires: pip install scapy
        """
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            raise ImportError("Install scapy: pip install scapy")

        print(f"[StreamBuilder] Reading PCAP {pcap_path} …")
        packets = rdpcap(pcap_path)
        rows = []
        for i, pkt in enumerate(packets):
            row = {
                "stream":           (pkt[TCP].sport + pkt[TCP].dport) if TCP in pkt else i // self.stream_window,
                "frame.len":        len(pkt),
                "frame.time_epoch": float(pkt.time),
                "ip.ttl":           pkt[IP].ttl if IP in pkt else 0,
                "payload":          len(pkt.payload),
                "srcport":          pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0),
                "dstport":          pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
                "tcp.seq_raw":      pkt[TCP].seq if TCP in pkt else 0,
                "tcp.ack_raw":      pkt[TCP].ack if TCP in pkt else 0,
                "tcp.window_size_value": pkt[TCP].window if TCP in pkt else 0,
                "tcp.flags.syn":    int(bool(TCP in pkt and pkt[TCP].flags & 0x02)),
                "tcp.flags.ack":    int(bool(TCP in pkt and pkt[TCP].flags & 0x10)),
                "tcp.flags.fin":    int(bool(TCP in pkt and pkt[TCP].flags & 0x01)),
                "tcp.flags.reset":  int(bool(TCP in pkt and pkt[TCP].flags & 0x04)),
                "tcp.flags.push":   int(bool(TCP in pkt and pkt[TCP].flags & 0x08)),
                "label":            label,
            }
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        raw_df["duration"] = 0.0  # will be computed per-stream in from_csv logic

        # Reuse grouping logic
        stream_records = []
        for _, grp in raw_df.groupby("stream"):
            grp = grp.copy()
            times = grp["frame.time_epoch"].sort_values()
            grp["duration"] = times.max() - times.min()
            rec = _build_stream_record(grp)
            rec["label"] = label
            stream_records.append(rec)

        return pd.DataFrame(stream_records).fillna(0.0)

    # ------------------------------------------------------------------
    @staticmethod
    def generate_synthetic(
        n_streams: int = 5000,
        attack_ratio: float = 0.4,
        random_state: int = 42,
        n_features: int = 58,
    ) -> pd.DataFrame:
        """
        Generate a synthetic stream dataset that mimics the paper's 58-field
        structure. Useful for quick testing without real capture data.

        Attack classes mirror the paper's confusion matrices:
            0 = Normal
            1 = ACK_Flooding
            2 = HTTP_Flooding
            3 = OS_Version_Detection
            4 = Port_Scanning
            5 = SYN_Flooding
            6 = Telnet_Bruteforce
        """
        rng = np.random.default_rng(random_state)

        CLASSES = [
            "Normal", "ACK_Flooding", "HTTP_Flooding",
            "OS_Version_Detection", "Port_Scanning",
            "SYN_Flooding", "Telnet_Bruteforce",
        ]
        n_attack   = int(n_streams * attack_ratio)
        n_normal   = n_streams - n_attack
        labels_idx = ([0] * n_normal +
                      list(rng.integers(1, len(CLASSES), size=n_attack)))
        rng.shuffle(labels_idx)

        # Base feature matrix: different distributions per class
        X = np.zeros((n_streams, n_features))
        for i, cls in enumerate(labels_idx):
            base  = rng.normal(loc=cls * 0.15, scale=0.1, size=n_features)
            noise = rng.normal(0, 0.05, size=n_features)
            # Inject class-specific signals
            if cls == 1:   # ACK Flooding: high ACK flag, high prate
                base[8]  += 0.9   # tcp.flags.ack.mean
                base[-1] += 0.8   # prate
            elif cls == 2: # HTTP Flooding: large frames
                base[13] += 0.7   # frame.len.mean
            elif cls == 4: # Port Scanning: high dstport std
                base[22] += 0.9   # dstport.std
            elif cls == 5: # SYN Flooding: high SYN flag, low interval std
                base[11] += 0.9   # tcp.flags.syn.mean
                base[35] -= 0.4   # int.std  (very regular intervals)
            X[i] = np.clip(base + noise, 0, 1)

        feature_names = [f"feature_{j:02d}" for j in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = [CLASSES[i] for i in labels_idx]
        print(f"[StreamBuilder] Synthetic dataset: {n_streams:,} streams, "
              f"{n_features} features, {len(CLASSES)} classes.")
        return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Build stream dataset")
    parser.add_argument("--input",  help="Path to input CSV or PCAP file")
    parser.add_argument("--output", default="streams.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset (no input needed)")
    parser.add_argument("--n-streams", type=int, default=5000)
    args = parser.parse_args()

    sb = StreamBuilder()
    if args.synthetic or not args.input:
        df = sb.generate_synthetic(n_streams=args.n_streams)
    elif args.input.endswith(".pcap"):
        df = sb.from_pcap(args.input)
    else:
        df = sb.from_csv(args.input)

    df.to_csv(args.output, index=False)
    print(f"[StreamBuilder] Saved → {args.output}")
