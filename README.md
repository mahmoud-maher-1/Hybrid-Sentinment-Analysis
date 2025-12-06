# 🎬 Movie Review Sentiment Analysis Suite

## Authors  
- **Akram ElNahtawy**  
- **Bassam Hassan**  
- **Mahmoud Maher**  
- **Mai Farahat**  
- **Mohanad Sabry**

---

## Try our live deployment at https://hybrid-sentiment-analysis.streamlit.app/

---
## 1\. Project Overview

This project is a comprehensive Deep Learning pipeline designed to classify movie reviews as either **Positive** or **Negative**. It leverages advanced Natural Language Processing (NLP) techniques and Recurrent Neural Networks (RNNs) to understand the context of text data.

The system is built to compare the performance of four distinct neural network architectures: **LSTM**, **Bi-LSTM**, **GRU**, and **Bi-GRU**. It includes a full lifecycle pipeline: from data ingestion and augmentation to training, evaluation, and deployment via a web interface.

-----

## 2\. Directory Structure

The project is organized modularly to separate concerns between data processing, modeling, and application logic.

  * **`main.py`**: The entry point. Orchestrates the entire training pipeline.
  * **`data_preprocessing.py`**: Handles text cleaning, augmentation, and vectorization.
  * **`model_training.py`**: Contains neural network architectures and callback configurations.
  * **`evaluation.py`**: Calculates metrics (accuracy, confusion matrices).
  * **`visualizations.py`**: Generates training graphs and comparison charts.
  * **`app.py`**: A Streamlit web application for real-time user inference.

-----

## 3\. Data Pipeline & Preprocessing

Before training, raw text data undergoes a rigorous transformation process to ensure high data quality.

### A. Text Cleaning

Raw text is noisy. The `clean_text` function normalizes the data through several steps:

  * **Normalization:** Converts text to lowercase and removes numbers/ordinals (e.g., "1st", "2nd").
  * **Filtering:** Removes URLs, HTML tags, and punctuation.
  * **Entity Removal:** uniquely removes specific month names and country names (using `pycountry`) to prevent the model from learning geographical or temporal biases.
  * **Lemmatization:** Converts words to their base root (e.g., "running" → "run") using NLTK.

### B. Data Augmentation

To make the models more robust, the pipeline synthetically expands the training dataset using three techniques:

1.  **Synonym Replacement:** Replaces words with their WordNet synonyms.
2.  **Random Swap:** Swaps the positions of two words in a sentence.
3.  **Random Deletion:** Randomly removes words with a probability of 15%.

### C. Vectorization & Balancing

  * **Tokenization:** Converts text into integer sequences (Vocabulary size: 20,000 words).
  * **SMOTE:** Applies **Synthetic Minority Over-sampling Technique** to the vectorized data to handle class imbalances, ensuring the model doesn't favor the majority class.

-----

## 4\. Model Architectures

This project implements four variations of RNNs using TensorFlow/Keras. All models use an **Embedding Layer** as the first input layer.

### 1\. LSTM (Long Short-Term Memory)

Standard LSTM designed to capture long-term dependencies in text.

  * **Structure:** Embedding $\rightarrow$ Conv1D (for feature extraction) $\rightarrow$ LSTM $\rightarrow$ Dense Layers.
  * **Best for:** General sequence modeling where context from the beginning of the sentence is needed at the end.

### 2\. Bi-LSTM (Bidirectional LSTM)

A more complex model that reads text both forwards and backwards.

  * **Structure:** Includes `SpatialDropout1D`, `Conv1D`, and two stacked `Bidirectional(LSTM)` layers. It utilizes both Average and Max pooling before the final classification.
  * **Best for:** Complex sentences where context depends on future words (e.g., sarcasm).

### 3\. GRU (Gated Recurrent Unit)

A streamlined version of LSTM that is computationally more efficient.

  * **Structure:** A simpler stack of GRU layers followed by Dropout and Dense layers.
  * **Best for:** Faster training with performance often comparable to LSTM.

### 4\. Bi-GRU

  * **Structure:** Combines Bidirectional wrappers around GRU layers with `SpatialDropout1D` to prevent overfitting.

-----

## 5\. Training Strategy

The training process is automated in `main.py` and highly tuned using **Callbacks**.

  * **Optimizer:** Adam Optimizer is used across models.
  * **EarlyStopping:** Stops training automatically if the validation loss stops improving (patience varies by model, typically 3-5 epochs).
  * **ModelCheckpoint:** Always saves the version of the model that achieved the highest validation accuracy, overwriting worse versions.
  * **ReduceLROnPlateau:** Reduces the learning rate by a factor of 0.5 if the model gets "stuck," allowing for fine-grained weight updates.

-----

## 6\. Visualization & Evaluation

The `evaluation.py` and `visualizations.py` modules generate artifacts to help understand model performance.

  * **Accuracy & Loss Plots:** Generates line graphs to detect overfitting (where training accuracy rises but validation accuracy drops).
  * **Confusion Matrix:** A heatmap showing how many True Positives vs. False Positives were generated.
  * **Model Comparison:** A bar chart comparing the final accuracy of all four models side-by-side.

-----

## 7\. Interactive Web Application

The project includes a **Streamlit** dashboard (`app.py`) for easy demonstration.

  * **Features:**
      * Allows users to select which model architecture (LSTM, Bi-LSTM, etc.) they want to use for prediction.
      * Accepts raw text input and outputs a "Positive" or "Negative" sentiment.
      * Displays a confidence score and a progress bar indicating how certain the model is.
  * **Note:** The app includes a standalone `clean_text` function to ensure deployed models receive data processed exactly the same way as during training.

-----

## 8\. How to Run

### Prerequisites

Ensure Python is installed along with the required libraries:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn nltk imbalanced-learn streamlit pycountry
```

### Training the Models

To run the full pipeline (cleaning, augmentation, training, and evaluation):

```bash
python main.py
```

*This will create a `Models/` directory containing the saved `.keras` files and a `Visualizations/` directory with performance graphs.*

### Launching the Dashboard

To start the user interface:

```bash
streamlit run app.py
```