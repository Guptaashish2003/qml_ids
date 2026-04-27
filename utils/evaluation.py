"""
Evaluation & Visualisation Utilities
=====================================
Reproduces the paper's evaluation outputs:
  - Confusion matrices  (Figures 5–8)
  - ROC curves          (Figures 5–8)
  - Training time table (Table 4)

Usage
-----
    from utils.evaluation import Evaluator
    ev = Evaluator(class_names=["Normal", "ACK_Flooding", ...])
    ev.plot_confusion(y_true, y_pred, title="QSVM")
    ev.plot_roc(X_test, y_test, model, title="QSVM")
    ev.benchmark_training_time(models_dict, X_sizes, X_full, y_full)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import label_binarize


class Evaluator:
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        save_dir: str = "results",
    ):
        self.class_names = class_names or [
            "ACK_Flooding", "HTTP_Flooding", "Normal",
            "OS_Version_Detection", "Port_Scanning",
            "SYN_Flooding", "Telnet_Bruteforce",
        ]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Model",
        normalise: bool = True,
        save: bool = True,
    ) -> np.ndarray:
        """Plot normalised confusion matrix (mirrors paper style)."""
        cm = confusion_matrix(y_true, y_pred,
                              labels=list(range(len(self.class_names))))
        if normalise:
            cm_plot = cm.astype(float)
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            cm_plot  = np.divide(cm_plot, row_sums,
                                 where=row_sums != 0, out=np.zeros_like(cm_plot))
        else:
            cm_plot = cm

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=".2f" if normalise else "d",
            cmap="YlOrRd",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            vmin=0, vmax=1 if normalise else None,
        )
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label",      fontsize=12)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save:
            path = self.save_dir / f"confusion_{title.replace(' ', '_')}.png"
            fig.savefig(path, dpi=150)
            print(f"[Evaluator] Saved → {path}")
        plt.close()
        return cm

    # ------------------------------------------------------------------
    def plot_roc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Model",
        save: bool = True,
    ):
        """
        Plot one-vs-rest ROC curves for each class.
        y_prob: (N, n_classes) probability matrix.
        """
        n_classes = len(self.class_names)
        y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, (cls_name, color) in enumerate(zip(self.class_names, colors)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.8,
                    label=f"ROC curve of class {i} (area = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curves — {title}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            path = self.save_dir / f"roc_{title.replace(' ', '_')}.png"
            fig.savefig(path, dpi=150)
            print(f"[Evaluator] Saved → {path}")
        plt.close()

    # ------------------------------------------------------------------
    def plot_combined(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Model",
        save: bool = True,
    ):
        """Side-by-side ROC + Confusion (matches paper's figure layout)."""
        n_classes = len(self.class_names)
        y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
        cm        = confusion_matrix(y_true, y_pred,
                                     labels=list(range(n_classes)))
        row_sums  = cm.sum(axis=1, keepdims=True)
        cm_norm   = np.divide(cm.astype(float), row_sums,
                              where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

        fig = plt.figure(figsize=(16, 6))
        gs  = gridspec.GridSpec(1, 2, figure=fig)

        # --- ROC ---
        ax0    = fig.add_subplot(gs[0])
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, (cls_name, color) in enumerate(zip(self.class_names, colors)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax0.plot(fpr, tpr, color=color, lw=1.5,
                     label=f"class {i} (area = {roc_auc:.2f})")
        ax0.plot([0, 1], [0, 1], "k--", lw=1)
        ax0.set_xlabel("False Positive Rate")
        ax0.set_ylabel("True Positive Rate")
        ax0.set_title(f"ROC Curves")
        ax0.legend(loc="lower right", fontsize=7)
        ax0.grid(alpha=0.3)

        # --- Confusion ---
        ax1 = fig.add_subplot(gs[1])
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=ax1, linewidths=0.5, linecolor="gray",
                    vmin=0, vmax=1)
        ax1.set_xlabel("Predicted label")
        ax1.set_ylabel("True label")
        ax1.set_title("Confusion matrix")
        plt.xticks(rotation=45, ha="right")

        fig.suptitle(f"Intrusion detection results — {title}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save:
            path = self.save_dir / f"combined_{title.replace(' ', '_')}.png"
            fig.savefig(path, dpi=150)
            print(f"[Evaluator] Saved → {path}")
        plt.close()

    # ------------------------------------------------------------------
    def benchmark_training_time(
        self,
        models: Dict,           # {"SVM": model_obj, "QSVM": model_obj, …}
        X_full: np.ndarray,
        y_full: np.ndarray,
        sizes: Optional[List[int]] = None,
        n_trials: int = 1,
        save: bool = True,
    ) -> dict:
        """
        Reproduce Table 4: measure training time vs dataset size.

        Parameters
        ----------
        models : dict name → unfitted model instance (with .fit() method)
        X_full, y_full : full dataset to subsample from
        sizes  : list of sample counts to benchmark
        """
        import time, copy
        from sklearn.model_selection import train_test_split

        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]

        results = {name: [] for name in models}

        for n in sizes:
            if n > len(X_full):
                n = len(X_full)
            idx   = np.random.choice(len(X_full), n, replace=False)
            X_sub = X_full[idx]
            y_sub = y_full[idx]
            for name, mdl in models.items():
                trial_times = []
                for _ in range(n_trials):
                    m_copy = copy.deepcopy(mdl)
                    t0 = time.time()
                    try:
                        m_copy.fit(X_sub, y_sub)
                        trial_times.append(time.time() - t0)
                    except Exception as e:
                        print(f"  [{name}] n={n} failed: {e}")
                        trial_times.append(np.nan)
                avg = np.nanmean(trial_times)
                results[name].append(avg)
                print(f"  {name:10s}  n={n:>7,}  time={avg:.2f}s")

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, times in results.items():
            ax.plot(sizes[:len(times)], times, marker="o", label=name)
        ax.set_xlabel("Input size (samples)")
        ax.set_ylabel("Training time (s)")
        ax.set_title("QML vs Classical ML — Training Time (Table 4)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            path = self.save_dir / "training_time_comparison.png"
            fig.savefig(path, dpi=150)
            print(f"[Evaluator] Saved → {path}")
        plt.close()

        return results

    # ------------------------------------------------------------------
    @staticmethod
    def print_table4(results: dict, sizes: List[int]):
        """Pretty-print a Table-4 style comparison."""
        headers = ["Input size"] + list(results.keys())
        row_fmt = "{:<15}" + "{:<22}" * len(results)
        print("\n" + "=" * (15 + 22 * len(results)))
        print(row_fmt.format(*headers))
        print("=" * (15 + 22 * len(results)))
        for i, n in enumerate(sizes):
            row = [f"{n:,}"] + [
                f"{results[name][i]:.2f}h" if not np.isnan(results[name][i])
                else "—"
                for name in results
            ]
            print(row_fmt.format(*row))
        print("=" * (15 + 22 * len(results)))
