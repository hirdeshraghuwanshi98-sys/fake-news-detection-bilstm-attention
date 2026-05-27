import re
import string
import joblib
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

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
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
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
    custom_objects={"AttentionLayer": AttentionLayer}
)

def predict_news(news_text):
    cleaned = clean_text(news_text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = model.predict(padded, verbose=0)[0][0]

    label = "Real News" if probability >= 0.5 else "Fake News"

    confidence = probability if probability >= 0.5 else (1 - probability)

    return {
        "label": label,
        "probability_real": float(probability),
        "confidence": float(confidence)
    }
