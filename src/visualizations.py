import matplotlib.pyplot as plt
import seaborn as sns
import os

VIZ_DIR = "../Visualizations"
if not os.path.exists(VIZ_DIR): os.makedirs(VIZ_DIR)


def plot_history(history, model_name):
    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"{model_name}_history.png"))
    plt.close()


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"{model_name}_cm.png"))
    plt.close()


def plot_comparison(results):
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.savefig(os.path.join(VIZ_DIR, "model_comparison.png"))
    plt.close()