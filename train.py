"""
train.py — Train the BiLSTM + Attention fake-news classifier.

Expected CSV columns (case-insensitive):  text / title / label
Label values accepted: fake / real  (or 0 / 1 where 1 = fake)

Usage
-----
    python train.py                         # uses data/news.csv by default
    python train.py --data data/myfile.csv  # custom path
    python train.py --epochs 10 --batch 64
"""

import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout,
    SpatialDropout1D, Layer,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_VOCAB  = 50_000
MAX_LEN    = 300
EMBED_DIM  = 128
LSTM_UNITS = 128
DROPOUT    = 0.3
EPOCHS     = 15
BATCH_SIZE = 32

MODEL_PATH     = os.path.join("models", "bilstm_attention_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
os.makedirs("models", exist_ok=True)

# ── NLTK setup ───────────────────────────────────────────────────────────────
nltk_dir = "/tmp/nltk_data"
os.makedirs(nltk_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_dir)
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, download_dir=nltk_dir, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


# ── Custom Attention layer ────────────────────────────────────────────────────
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight("att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("att_bias",   shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e      = K.tanh(K.dot(x, self.W) + self.b)
        a      = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        return super().get_config()


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(vocab_size: int) -> Model:
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN)(inp)
    x   = SpatialDropout1D(DROPOUT)(x)
    x   = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True,
                              dropout=DROPOUT, recurrent_dropout=0.1))(x)
    x   = Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=True,
                              dropout=DROPOUT, recurrent_dropout=0.1))(x)
    x   = AttentionLayer()(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(DROPOUT)(x)
    x   = Dense(64,  activation="relu")(x)
    x   = Dropout(DROPOUT / 2)(x)
    out = Dense(1,   activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ── Data loader ───────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    # Combine title + text if both present
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("CSV must have a 'text' column.")

    if "label" not in df.columns:
        raise ValueError("CSV must have a 'label' column (fake/real or 0/1).")

    label_map = {"fake": 1, "real": 0, "0": 0, "1": 1, 0: 0, 1: 1}
    df["label"] = df["label"].str.lower().map(label_map) if df["label"].dtype == object else df["label"].map(label_map)
    df = df.dropna(subset=["label", "content"])
    return df[["content", "label"]]


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_path: str, epochs: int, batch_size: int):
    print(f"[train] Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"[train] Samples: {len(df)}  |  Fake: {df['label'].sum()}  Real: {(df['label']==0).sum()}")

    print("[train] Cleaning text …")
    df["content"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"].values, df["label"].values,
        test_size=0.2, random_state=42, stratify=df["label"].values
    )

    print("[train] Fitting tokenizer …")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[train] Tokenizer saved → {TOKENIZER_PATH}")

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_tr = encode(X_train)
    X_te = encode(X_test)

    print("[train] Building model …")
    model = build_model(vocab_size)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    print("[train] Training …")
    model.fit(
        X_tr, y_train,
        validation_data=(X_te, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    print("\n[train] Evaluation on test set:")
    y_pred = (model.predict(X_te) >= 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"\n[train] Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=os.path.join("data", "news.csv"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch)
