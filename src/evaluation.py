import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} ---")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
    }