"""
Test saved QSVM & SVM models
=============================
Loads saved models from results/ and verifies they predict correctly.

Usage:
    python test_models.py
    python test_models.py --results results/
"""

import os
import sys
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def test_model(name: str, model_path: str, X_test, y_test):
    """Load a saved model and run predictions on test data."""
    print(f"\n{'─'*50}")
    print(f"  Testing: {name}")
    print(f"{'─'*50}")

    # 1. Load model
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model = joblib.load(model_path)
    print(f"  ✓ Loaded model from {model_path}")
    print(f"    Classes: {list(model.le.classes_)}")

    # 2. Predict
    y_pred = model.predict(X_test)
    y_true = model.le.transform(y_test)
    print(f"  ✓ Predicted {len(y_pred)} samples")

    # 3. Accuracy check
    acc = accuracy_score(y_true, y_pred)
    print(f"  ✓ Accuracy: {acc*100:.2f}%")

    # 4. Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=model.le.classes_,
        zero_division=0,
    )
    print(f"\n{report}")

    # 5. Sanity checks
    assert len(y_pred) == len(X_test), "Prediction count mismatch"
    assert acc > 0.0, "Accuracy is 0% — model is broken"
    n_classes_predicted = len(np.unique(y_pred))
    assert n_classes_predicted >= 2, \
        f"Model only predicts {n_classes_predicted} class(es) — likely degenerate"

    print(f"  ✓ All checks passed for {name}")
    return acc


def test_single_sample(name: str, model_path: str, X_test, y_test):
    """Test prediction on a single sample to verify inference works."""
    model = joblib.load(model_path)
    single_x = X_test[:1]
    pred = model.predict(single_x)
    label = model.le.inverse_transform(pred)[0]
    true_label = y_test[0]
    print(f"  [{name}] Single-sample prediction: {label}  (true: {true_label})")
    return pred


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test saved QML models")
    parser.add_argument("--results", default="results",
                        help="Directory containing saved models")
    args = parser.parse_args()

    results_dir = args.results

    print("=" * 50)
    print("  QML Model Test Suite")
    print("=" * 50)

    # Load test data
    test_data_path = os.path.join(results_dir, "test_data.joblib")
    assert os.path.exists(test_data_path), \
        f"Test data not found at {test_data_path}. Run main.py first."
    data = joblib.load(test_data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"\n  Loaded test data: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"  Classes in test set: {sorted(set(y_test))}")

    # Discover and test all saved models
    model_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith("_model.joblib")
    ])

    if not model_files:
        print(f"\n  ✗ No model files found in {results_dir}/")
        sys.exit(1)

    print(f"  Found {len(model_files)} saved model(s): {model_files}")

    all_results = {}
    all_passed = True

    for mf in model_files:
        name = mf.replace("_model.joblib", "").upper()
        model_path = os.path.join(results_dir, mf)
        try:
            acc = test_model(name, model_path, X_test, y_test)
            test_single_sample(name, model_path, X_test, y_test)
            all_results[name] = acc
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            all_passed = False

    # Final summary
    print("\n" + "=" * 50)
    print("  Test Summary")
    print("=" * 50)
    for name, acc in all_results.items():
        print(f"  ✓ {name:10s}  {acc*100:.2f}%")

    if all_passed:
        print(f"\n  ✅ All {len(model_files)} model(s) passed all tests!")
    else:
        print(f"\n  ❌ Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
