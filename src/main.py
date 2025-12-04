import os
import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data, get_vectors, apply_smote
from model_training import (
    build_lstm, build_bi_lstm, build_gru, build_bi_gru,
    train_model, get_callbacks
)
from evaluation import evaluate_model
from visualizations import plot_history, plot_confusion_matrix, plot_comparison

# Config
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128
EPOCHS = 15  # Increased default epochs since we now have EarlyStopping
BATCH_SIZE = 2048
TRAIN_PATH = "../Dataset/train_data.csv"
TEST_PATH = "../Dataset/test_data.csv"
MODEL_DIR = "../Models"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)


def main():
    # 1. Load, Augment (Text Level), and Clean
    print("1. Loading and Preprocessing Data...")
    if not os.path.exists(TRAIN_PATH):
        print("Dataset not found. Please place train.csv/test.csv in Dataset/ folder.")
        return

    train_df, test_df, label_col = load_and_preprocess_data(TRAIN_PATH, TEST_PATH, apply_aug=True)

    # 2. Vectorize
    print("2. Vectorizing...")
    X_train_raw, X_test, tokenizer = get_vectors(
        train_df['clean_text'],
        test_df['clean_text'],
        MAX_WORDS,
        MAX_LEN,
        save_tokenizer_path=os.path.join(MODEL_DIR, "tokenizer.pickle")
    )
    y_train_raw = train_df[label_col].values
    y_test = test_df[label_col].values

    # 3. Apply SMOTE (Vector Level)
    X_train, y_train = apply_smote(X_train_raw, y_train_raw)

    # 4. Define Models
    models_to_train = {
        "lstm": build_lstm(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "bi-lstm": build_bi_lstm(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "gru": build_gru(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "bi-gru": build_bi_gru(MAX_WORDS, EMBED_DIM, MAX_LEN)
    }

    results = {}

    # 5. Train Loop
    print("3. Starting Training Loop...")
    for name, model in models_to_train.items():
        print(f"\n--- Training {name.upper()} ---")

        # Get specific callbacks for this model
        callbacks = get_callbacks(name, model_dir=MODEL_DIR)

        history = train_model(
            model,
            X_train, y_train,
            X_test, y_test,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks  # Pass the retrieved callbacks (or None)
        )

        # Save Final Model (Note: Checkpoints might also be saved in Models/ by callbacks)
        final_save_path = os.path.join(MODEL_DIR, f"{name}.keras")
        model.save(final_save_path)
        print(f"Final model saved to {final_save_path}")

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics

        # Plot
        plot_history(history, name)
        plot_confusion_matrix(metrics['confusion_matrix'], name)

    # Final Comparison
    plot_comparison(results)
    print("\nPipeline Complete. Artifacts saved.")


if __name__ == "__main__":
    main()