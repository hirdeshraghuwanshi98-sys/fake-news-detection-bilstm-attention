import os, re, pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MODEL_PATH     = os.path.join("models", "bilstm_attention_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MAX_LEN        = 500

_nltk_ready = False
def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    d = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(d, exist_ok=True)
    nltk.data.path.insert(0, d)
    for p in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(p, download_dir=d, quiet=True)
    _nltk_ready = True

def clean_text(text):
    _ensure_nltk()
    lem  = WordNetLemmatizer()
    stop = set(stopwords.words("english"))
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(lem.lemmatize(w) for w in text.split()
                    if w not in stop and len(w) > 2)

_model     = None
_tokenizer = None

def _load_artifacts():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Model not found. Ensure '{MODEL_PATH}' and '{TOKENIZER_PATH}' exist."
        )

    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, LSTM, Dense,
        Dropout, SpatialDropout1D, Layer
    )
    import tensorflow.keras.backend as K
    import h5py

    class AttentionLayer(Layer):
        def build(self, input_shape):
            self.W = self.add_weight("att_weight",
                shape=(input_shape[-1], 1), initializer="glorot_uniform")
            self.b = self.add_weight("att_bias",
                shape=(input_shape[1], 1), initializer="zeros")
            super().build(input_shape)
        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            return K.sum(x * a, axis=1)
        def get_config(self):
            return super().get_config()

    # Read layer names directly from the .h5 file
    with h5py.File(MODEL_PATH, "r") as f:
        if "model_weights" in f:
            saved_layer_names = list(f["model_weights"].keys())
        elif "layer_names" in f.attrs:
            saved_layer_names = [n.decode() if isinstance(n, bytes) else n
                                 for n in f.attrs["layer_names"]]
        else:
            saved_layer_names = []

    # Try full model load first
    try:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
        return _model, _tokenizer
    except Exception:
        pass

    # Build skeleton matching saved layer names exactly
    has_spatial = any("spatial" in n.lower() for n in saved_layer_names)

    inp = Input(shape=(MAX_LEN,), name="input_layer")
    x   = Embedding(50000, 128, name="embedding")(inp)
    if has_spatial:
        x = SpatialDropout1D(0.2, name="spatial_dropout1d")(x)
    x   = Bidirectional(LSTM(64, return_sequences=True), name="bidirectional")(x)
    x   = AttentionLayer(name="attention_layer")(x)
    x   = Dense(64, activation="relu", name="dense")(x)
    x   = Dropout(0.5, name="dropout")(x)
    out = Dense(1, activation="sigmoid", name="dense_1")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    _model = model

    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)
    return _model, _tokenizer


def predict(text: str) -> dict:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tokenizer = _load_artifacts()
    cleaned   = clean_text(text)
    seq       = tokenizer.texts_to_sequences([cleaned])
    padded    = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob_fake = float(model.predict(padded, verbose=0)[0][0])
    prob_real = 1.0 - prob_fake
    label     = "FAKE" if prob_fake >= 0.5 else "REAL"
    return {
        "label":        label,
        "confidence":   prob_fake if label == "FAKE" else prob_real,
        "fake_prob":    prob_fake,
        "real_prob":    prob_real,
        "cleaned_text": cleaned,
    }

def predict_news(text: str) -> dict:
    return predict(text)
