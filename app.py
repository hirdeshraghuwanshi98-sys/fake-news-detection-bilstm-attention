"""
app.py — Fake News Detection · Streamlit Web App
Production-ready UI with BiLSTM + Attention model inference.
"""

import os
import sys
import time
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="FakeScope · Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* Background */
.stApp { background: #0a0e1a; color: #e8eaf6; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e2a40;
}
[data-testid="stSidebar"] * { color: #c9d1e8 !important; }

/* Cards */
.card {
    background: linear-gradient(135deg, #111827 0%, #0d1321 100%);
    border: 1px solid #1e2a40;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

/* Result pill */
.result-fake {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff4d6d, #c9184a);
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1.5px;
}
.result-real {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 999px;
    background: linear-gradient(90deg, #06d6a0, #0cb89b);
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1.5px;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1e2a40;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin: 8px 0;
}
.conf-bar-fill-fake {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff4d6d, #c9184a);
    transition: width 0.8s ease;
}
.conf-bar-fill-real {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #06d6a0, #0cb89b);
    transition: width 0.8s ease;
}

/* Text area */
textarea {
    background: #111827 !important;
    color: #e8eaf6 !important;
    border: 1px solid #1e2a40 !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 10px;
    padding: 12px 0;
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
div[data-testid="stButton"]:first-child > button {
    background: linear-gradient(90deg, #4f8ef7, #1e5fd4);
    color: #fff;
}
div[data-testid="stButton"]:first-child > button:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #4f8ef7 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Horizontal rule */
hr { border-color: #1e2a40; }

/* Expander */
details { background: #111827; border: 1px solid #1e2a40; border-radius: 10px; }

/* Info / warning */
.stAlert { border-radius: 10px !important; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("models", "bilstm_attention_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MODEL_READY    = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

EXAMPLES = [
    {
        "title": "Political Propaganda",
        "text": (
            "BREAKING: Scientists CONFIRM that 5G towers are being used by the government "
            "to implant tracking microchips in COVID vaccines. Whistleblowers reveal this "
            "shocking truth that mainstream media refuses to report. Share before it's deleted!"
        ),
    },
    {
        "title": "Legitimate News",
        "text": (
            "The Federal Reserve raised its benchmark interest rate by 25 basis points on "
            "Wednesday, bringing it to the highest level in 22 years. The move was widely "
            "anticipated by markets and reflects the central bank's continued efforts to "
            "bring inflation back toward its 2% target."
        ),
    },
    {
        "title": "Clickbait Misinformation",
        "text": (
            "SHOCKING: This one fruit DESTROYS cancer cells overnight! Big Pharma has been "
            "hiding this cure for decades. Doctors HATE this simple trick that cures all "
            "diseases. Click here to see what they don't want you to know!"
        ),
    },
]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load model artifacts once and cache them."""
    from predict import _load_artifacts
    return _load_artifacts()


def run_prediction(text: str) -> dict:
    from predict import predict
    return predict(text)


def confidence_bar(value: float, label: str) -> str:
    pct   = int(value * 100)
    color = "fake" if label == "FAKE" else "real"
    return f"""
    <div style="margin:4px 0 12px 0">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8892b0;margin-bottom:4px">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill-{color}" style="width:{pct}%"></div>
      </div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FakeScope")
    st.markdown("*AI-powered fake news detector*")
    st.markdown("---")

    st.markdown("### Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Architecture", "BiLSTM")
    with col2:
        st.metric("Mechanism", "Attention")

    st.markdown("")
    st.metric("Max Tokens", "300")
    st.metric("Vocab Size", "50 K")

    st.markdown("---")
    st.markdown("### Quick Examples")
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"📄 {ex['title']}", key=f"ex_{i}"):
            st.session_state["input_text"] = ex["text"]

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
1. **NLP Preprocessing** — tokenize, lemmatize, remove stopwords
2. **Embedding Layer** — maps tokens to dense vectors
3. **BiLSTM × 2** — captures bidirectional context
4. **Attention** — weights important phrases
5. **Dense classifier** — outputs Fake / Real probability
    """)

    st.markdown("---")
    st.caption("Built with TensorFlow · Keras · Streamlit")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 8px 0">
  <div style="font-size:3rem;margin-bottom:8px">🔍</div>
  <h1 style="font-size:2.4rem;font-weight:800;
             background:linear-gradient(90deg,#4f8ef7,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:0">FakeScope</h1>
  <p style="color:#8892b0;font-size:1.05rem;margin-top:6px">
    Detect fake news using BiLSTM + Attention deep learning
  </p>
</div>
""", unsafe_allow_html=True)

# Model status banner
if not MODEL_READY:
    st.warning(
        "⚠️ **Trained model not found.** "
        "Run `python train.py` locally with your dataset to generate "
        "`models/bilstm_attention_model.h5` and `models/tokenizer.pkl`, "
        "then push them to your repository.",
        icon="⚠️",
    )
else:
    with st.spinner("Loading model…"):
        try:
            load_model()
            st.success("✅ Model loaded and ready!", icon="✅")
        except Exception as e:
            st.error(f"Model load error: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📝 Enter News Article")

input_text = st.text_area(
    label="Paste article text or headline below",
    value=st.session_state.get("input_text", ""),
    height=180,
    placeholder="Paste a news article, headline, or social media post here…",
    key="main_input",
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([3, 1, 1])
with col_a:
    analyze_btn = st.button("🔍 Analyze Article", disabled=not MODEL_READY, use_container_width=True)
with col_b:
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.metric("Words", word_count)
with col_c:
    char_count = len(input_text)
    st.metric("Chars", char_count)

st.markdown("</div>", unsafe_allow_html=True)


# ── Prediction output ─────────────────────────────────────────────────────────
if analyze_btn:
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    elif len(input_text.split()) < 5:
        st.warning("Please enter at least 5 words for a reliable prediction.")
    else:
        with st.spinner("Analyzing with BiLSTM + Attention model…"):
            t0 = time.time()
            try:
                result = run_prediction(input_text)
                elapsed = time.time() - t0

                label      = result["label"]
                confidence = result["confidence"]
                fake_prob  = result["fake_prob"]
                real_prob  = result["real_prob"]
                cleaned    = result["cleaned_text"]

                # ── Result card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Analysis Result")

                r1, r2, r3 = st.columns([2, 2, 1])
                with r1:
                    pill_cls = "result-fake" if label == "FAKE" else "result-real"
                    icon     = "🚨" if label == "FAKE" else "✅"
                    st.markdown(
                        f'<p style="color:#8892b0;font-size:0.85rem;margin-bottom:6px">VERDICT</p>'
                        f'<span class="{pill_cls}">{icon} {label}</span>',
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with r3:
                    st.metric("Inference", f"{elapsed*1000:.0f} ms")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probability breakdown**")
                st.markdown(confidence_bar(fake_prob, "FAKE"), unsafe_allow_html=True)
                st.markdown(confidence_bar(real_prob, "REAL"), unsafe_allow_html=True)

                # ── Interpretation
                st.markdown("---")
                if label == "FAKE":
                    if confidence >= 0.90:
                        msg = "🚨 **Very High Risk** — This article exhibits strong patterns commonly associated with fake or misleading news."
                    elif confidence >= 0.75:
                        msg = "⚠️ **High Risk** — Several indicators suggest this may be unreliable. Cross-check with trusted sources."
                    else:
                        msg = "🟡 **Moderate Risk** — Some misleading patterns detected. Verify with additional sources."
                else:
                    if confidence >= 0.90:
                        msg = "✅ **Likely Credible** — Strong indicators of factual, well-structured reporting."
                    elif confidence >= 0.75:
                        msg = "🟢 **Probably Credible** — Appears to follow patterns of legitimate news writing."
                    else:
                        msg = "🟡 **Uncertain** — Cannot determine with high confidence. Always verify independently."

                st.info(msg)

                with st.expander("🔬 View preprocessed text"):
                    st.code(cleaned, language=None)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Disclaimer
                st.caption(
                    "⚠️ This tool is for educational purposes. "
                    "Always verify news from authoritative sources such as Reuters, AP, BBC, or government websites."
                )

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)


# ── Batch analysis ────────────────────────────────────────────────────────────
with st.expander("📂 Batch Analysis (upload CSV)"):
    st.markdown("Upload a CSV with a **`text`** column to analyze multiple articles at once.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded and MODEL_READY:
        import pandas as pd
        df_up = pd.read_csv(uploaded)
        if "text" not in df_up.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.write(f"Loaded **{len(df_up)}** articles.")
            if st.button("🚀 Run Batch Analysis"):
                prog = st.progress(0)
                results = []
                for i, row in df_up.iterrows():
                    try:
                        r = run_prediction(str(row["text"]))
                        results.append({
                            "text":       str(row["text"])[:80] + "…",
                            "label":      r["label"],
                            "confidence": f"{r['confidence']*100:.1f}%",
                            "fake_prob":  f"{r['fake_prob']*100:.1f}%",
                            "real_prob":  f"{r['real_prob']*100:.1f}%",
                        })
                    except Exception:
                        results.append({"text": str(row["text"])[:80], "label": "ERROR",
                                        "confidence": "-", "fake_prob": "-", "real_prob": "-"})
                    prog.progress((i + 1) / len(df_up))

                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True)
                csv_out = df_res.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", csv_out,
                                   "fakescope_results.csv", "text/csv")


# ── About section ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown("""
### Fake News Detection using BiLSTM + Attention

This project uses a **Bidirectional LSTM with Attention Mechanism** to classify
news articles as real or fake.

| Component | Details |
|---|---|
| Embedding | Trainable word embeddings (128-dim) |
| Encoder | 2-layer BiLSTM (128 + 64 units each direction) |
| Attention | Bahdanau-style soft attention |
| Classifier | Dense(128) → Dense(64) → Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam with LR scheduling |

**Preprocessing pipeline:** lowercase → URL/HTML removal → punctuation stripping →
stopword removal → WordNet lemmatization → tokenization → padding.

**Training data:** LIAR dataset / WELFake / custom labelled corpus.
    """)
    
import streamlit as st
try:
    from predict import predict_news

except Exception as e:

    import streamlit as st

    st.error(f"Model loading error: {e}")

    st.stop()

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection Using BiLSTM + Attention")

st.write(
    "Paste a news article below and the model will predict whether it is fake or real."
)

news_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article text here..."
)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        result = predict_news(news_text)

        label = result["label"]
        confidence = result["confidence"] * 100
        probability_real = result["probability_real"] * 100

        if label == "Real News":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Probability Real", f"{probability_real:.2f}%")

st.markdown("---")
st.caption("Developed by Hirdesh Raghuwanshi")
