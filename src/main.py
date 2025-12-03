import os
import pandas as pd
import numpy as np
from data_preprocessing import load_and_clean_data, fit_tokenizer, preprocess_sequences
from model_training import build_lstm, build_bi_lstm, build_gru, build_bi_gru, train_model
from evaluation import evaluate_model
from visualizations import plot_history, plot_confusion_matrix, plot_comparison

# Configuration
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128
EPOCHS = 15
BATCH_SIZE = 32
TRAIN_PATH = "Dataset/train_data.csv"
TEST_PATH = "Dataset/test_data.csv"
MODEL_DIR = "Models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def main():
    print("1. Loading and Preprocessing Data...")

    train_df, test_df, label_col = load_and_clean_data(TRAIN_PATH, TEST_PATH)

    # Tokenizer
    tokenizer = fit_tokenizer(train_df['clean_text'], MAX_WORDS, save_path=os.path.join(MODEL_DIR, "tokenizer.pickle"))

    # Vectors
    X_train = preprocess_sequences(tokenizer, train_df['clean_text'], MAX_LEN)
    y_train = train_df[label_col].values
    X_test = preprocess_sequences(tokenizer, test_df['clean_text'], MAX_LEN)
    y_test = test_df[label_col].values

    # Model Definitions
    models_to_train = {
        "lstm": build_lstm(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "bi-lstm": build_bi_lstm(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "gru": build_gru(MAX_WORDS, EMBED_DIM, MAX_LEN),
        "bi-gru": build_bi_gru(MAX_WORDS, EMBED_DIM, MAX_LEN)
    }

    results = {}

    print("2. Starting Training Loop...")
    for name, model in models_to_train.items():
        print(f"\nTraining {name.upper()}...")
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Save Model
        model.save(os.path.join(MODEL_DIR, f"{name}.h5"))

        # Evaluate
        eval_metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = eval_metrics

        # Visualizations
        plot_history(history, name)
        plot_confusion_matrix(eval_metrics['confusion_matrix'], name)

    # Final Comparison Plot
    plot_comparison(results)
    print("\nPipeline Complete. Models and Visualizations saved.")


if __name__ == "__main__":
    main()