"""
predict.py — Fake News Detection inference module
Loads the trained BiLSTM+Attention model and tokenizer, returns predictions.
"""

import os
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK data (downloaded once, cached in /tmp on Streamlit Cloud) ──────────
_NLTK_READY = False

def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    nltk_dir = "/tmp/nltk_data"
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_dir)
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, download_dir=nltk_dir, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("models", "bilstm_attention_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MAX_LEN        = 300   # must match training


# ── Text preprocessing ───────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation, remove stopwords, lemmatize."""
    _ensure_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URLs
    text = re.sub(r"<.*?>", "", text)                       # HTML tags
    text = re.sub(r"[^a-z\s]", "", text)                   # non-alpha
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


# ── Model / tokenizer loader (cached) ────────────────────────────────────────
_model     = None
_tokenizer = None

def _load_artifacts():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py first to generate "
            f"'{MODEL_PATH}' and '{TOKENIZER_PATH}'."
        )

    # Lazy import so the app doesn't crash if TF isn't installed yet
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Layer
    import tensorflow.keras.backend as K

    class AttentionLayer(Layer):
        """Custom Bahdanau-style attention used during training."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1], 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b = self.add_weight(
                name="att_bias",
                shape=(input_shape[1], 1),
                initializer="zeros",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

        def get_config(self):
            return super().get_config()

    _model = load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    return _model, _tokenizer


# ── Public API ────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Predict whether *text* is Fake or Real news.

    Returns
    -------
    dict with keys:
        label       : "FAKE" | "REAL"
        confidence  : float 0–1  (confidence in predicted label)
        fake_prob   : float 0–1
        real_prob   : float 0–1
        cleaned_text: str
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer = _load_artifacts()

    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    prob_fake = float(model.predict(padded, verbose=0)[0][0])
    prob_real = 1.0 - prob_fake

    label      = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else prob_real

    return {
        "label":        label,
        "confidence":   confidence,
        "fake_prob":    prob_fake,
        "real_prob":    prob_real,
        "cleaned_text": cleaned,

import re
import string
import joblib
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from keras.models import load_model
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer


nltk.download('stopwords')

MAX_LENGTH = 500

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


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

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = attention_weights * inputs

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(words)


TOKENIZER_PATH = "models/tokenizer.pkl"

MODEL_PATH = "models/fake_news_bilstm_attention.keras"

tokenizer = joblib.load(TOKENIZER_PATH)

model = load_model(
    MODEL_PATH,
    custom_objects={
        "AttentionLayer": AttentionLayer
    },
    compile=False
)


def predict_news(news_text):

    cleaned = clean_text(news_text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    label = (
        "Real News"
        if probability >= 0.5
        else "Fake News"
    )

    confidence = (
        probability
        if probability >= 0.5
        else (1 - probability)
    )

    return {
        "label": label,
        "probability_real": float(probability),
        "confidence": float(confidence)
    }
