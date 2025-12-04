import re
import string
import random
import pandas as pd
import numpy as np
import pickle
import nltk
import pycountry
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle


# Ensure NLTK resources are available
def download_nltk_resources():
    resources = ["wordnet", "omw-1.4", "stopwords", "punkt", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)


download_nltk_resources()


# --- Augmentation Functions ---
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            if lem.name().lower() != word.lower():
                synonyms.add(lem.name().replace("_", " "))
    return list(synonyms)


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    if len(words) < 2: return sentence
    candidates = [w for w in words if get_synonyms(w)]
    if not candidates: return sentence
    for _ in range(n):
        word = random.choice(candidates)
        synonym = random.choice(get_synonyms(word))
        words = [synonym if w == word else w for w in words]
    return " ".join(words)


def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1: return sentence
    new_words = [w for w in words if random.random() > p]
    return " ".join(new_words) if new_words else random.choice(words)


def random_swap(sentence, n=1):
    words = sentence.split()
    if len(words) < 2: return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def augment_text(text):
    choice = random.choice(["synonym", "swap", "delete"])
    if choice == "synonym":
        return synonym_replacement(text)
    elif choice == "swap":
        return random_swap(text)
    elif choice == "delete":
        return random_deletion(text, p=0.15)
    return text


def apply_text_augmentation(df, text_col, label_col):
    print("Applying text augmentation...")
    augmented_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        new_text = augment_text(row[text_col])
        augmented_rows.append({text_col: new_text, label_col: row[label_col]})

    aug_df = pd.DataFrame(augmented_rows)
    # Concatenate and shuffle
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    return shuffle(combined_df, random_state=42).reset_index(drop=True)


# --- Cleaning Functions ---
def clean_text(text):
    # Initialize Lemmatizer & Stopwords inside to ensure resources are loaded
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 1. Lowercase
    text = text.lower()

    # 2. Remove numbers and specific patterns
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', '', text)  # Ordinals
    # Remove months
    months = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    text = re.sub(months, '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove Countries
    for c in pycountry.countries:
        if c.name.lower() in text:
            text = re.sub(r'\b' + re.escape(c.name.lower()) + r'\b', '', text)

    # 4. Tokenize, Stopwords, Lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)


# --- Pipeline Functions ---
def load_and_preprocess_data(train_path, test_path, apply_aug=True):
    train_df = pd.read_csv(train_path)
    train_df.drop_duplicates(inplace=True)
    train_df.sample(frac=1).reset_index(drop=True)
    test_df = pd.read_csv(test_path)
    test_df.drop_duplicates(inplace=True)
    test_df.sample(frac=1).reset_index(drop=True)

    # Assume cols are [text, label]
    text_col = train_df.columns[0]
    label_col = train_df.columns[1]

    # Apply Augmentation to Train only
    if apply_aug:
        train_df = apply_text_augmentation(train_df, text_col, label_col)

    print("Cleaning text (this may take a while due to pycountry/lemmatization)...")
    train_df['clean_text'] = train_df[text_col].apply(clean_text)
    test_df['clean_text'] = test_df[text_col].apply(clean_text)

    return train_df, test_df, label_col


def get_vectors(train_texts, test_texts, max_words, max_len, save_tokenizer_path=None):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post")
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len, padding="post")

    if save_tokenizer_path:
        with open(save_tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, X_test, tokenizer


def apply_smote(X, y):
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res