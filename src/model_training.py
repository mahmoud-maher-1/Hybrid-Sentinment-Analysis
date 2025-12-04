import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Embedding, LSTM, GRU, Dense, Dropout,
                                     Bidirectional, Conv1D, GlobalMaxPooling1D,
                                     SpatialDropout1D, GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def get_callbacks(model_name, model_dir="../Models"):
    """
    Returns the specific callbacks used in the reference notebooks for each model.
    Paths are adjusted to save inside the 'Models/' directory.
    """
    callbacks = []

    # 1. BI-LSTM Callbacks (from BI-LSTM.py)
    if model_name == "bi-lstm":
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(model_dir, "best_cnn_lstm_model.keras"),
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                verbose=1
            )
        ]

    # 2. LSTM Callbacks (from LSTM.ipynb)
    elif model_name == "lstm":
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(model_dir, "best_cnn_lstm_model.keras"),
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]

    # 3. BI-GRU Callbacks (from BI-GRU.ipynb)
    elif model_name == "bi-gru":
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(model_dir, "best_bigru_model.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

    # 4. GRU Callbacks (from GRU.ipynb - No callbacks used)
    elif model_name == "gru":
        return None

    return callbacks


# --- Model Architectures (Unchanged) ---

def build_lstm(vocab_size, embed_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len, mask_zero=True),
        Conv1D(64, 5, activation="relu", kernel_regularizer=l2(0.002)),
        Dropout(0.5),
        LSTM(32, return_sequences=True, kernel_regularizer=l2(0.002)),
        Dropout(0.5),
        GlobalMaxPooling1D(),
        Dense(16, activation="relu", kernel_regularizer=l2(0.002)),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
    return model


def build_gru(vocab_size, embed_dim, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    return model


def build_bi_gru(vocab_size, embed_dim, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.0)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_bi_lstm(vocab_size, embed_dim, max_len):
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embed_dim, input_length=max_len)(inputs)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(128, 5, padding="same", activation="relu", kernel_regularizer=l2(1e-3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.4))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=3e-4), metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, callbacks=None):
    """
    Trains the model with optional callbacks.
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history