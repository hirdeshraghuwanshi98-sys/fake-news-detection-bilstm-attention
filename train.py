
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import tensorflow as tf
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

nltk.download('stopwords')

print("Loading dataset...")

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\\s+', ' ', text).strip()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(words)

print("Cleaning text...")

df["cleaned_content"] = df["content"].apply(clean_text)

MAX_WORDS = 50000
MAX_LENGTH = 500

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["cleaned_content"])

sequences = tokenizer.texts_to_sequences(df["cleaned_content"])

X = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

class AttentionLayer(Layer):

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * inputs

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

VOCAB_SIZE = 50000
EMBEDDING_DIM = 128
LSTM_UNITS = 64

inputs = Input(shape=(MAX_LENGTH,))

x = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM
)(inputs)

x = Bidirectional(
    LSTM(
        LSTM_UNITS,
        return_sequences=True
    )
)(x)

x = AttentionLayer()(x)

x = Dense(64, activation="relu")(x)

x = Dropout(0.5)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/fake_news_bilstm_attention.keras"

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

print("Training model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

joblib.dump(tokenizer, "models/tokenizer.pkl")

print("Evaluating model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)

loss, accuracy = best_model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

y_pred_prob = best_model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"]
))

print("Training completed successfully!")
