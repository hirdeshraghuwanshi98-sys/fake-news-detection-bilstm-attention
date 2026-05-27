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