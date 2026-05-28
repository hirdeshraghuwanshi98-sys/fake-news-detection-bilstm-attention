import streamlit as st

# ── MUST be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="FakeScope · Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os, time

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf6; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1321 0%,#111827 100%); border-right:1px solid #1e2a40; }
[data-testid="stSidebar"] * { color: #c9d1e8 !important; }
.card { background:linear-gradient(135deg,#111827 0%,#0d1321 100%); border:1px solid #1e2a40; border-radius:16px; padding:24px 28px; margin-bottom:16px; }
.result-fake { display:inline-block; padding:6px 22px; border-radius:999px; background:linear-gradient(90deg,#ff4d6d,#c9184a); color:#fff; font-size:1.05rem; font-weight:700; letter-spacing:1.5px; }
.result-real { display:inline-block; padding:6px 22px; border-radius:999px; background:linear-gradient(90deg,#06d6a0,#0cb89b); color:#fff; font-size:1.05rem; font-weight:700; letter-spacing:1.5px; }
.conf-bar-bg { background:#1e2a40; border-radius:999px; height:10px; width:100%; margin:8px 0; }
.conf-bar-fake { height:10px; border-radius:999px; background:linear-gradient(90deg,#ff4d6d,#c9184a); }
.conf-bar-real { height:10px; border-radius:999px; background:linear-gradient(90deg,#06d6a0,#0cb89b); }
textarea { background:#111827 !important; color:#e8eaf6 !important; border:1px solid #1e2a40 !important; border-radius:10px !important; }
.stButton > button { width:100%; border:none; border-radius:10px; padding:12px 0; font-weight:700; background:linear-gradient(90deg,#4f8ef7,#1e5fd4); color:#fff; }
.stButton > button:hover { filter:brightness(1.12); }
[data-testid="stMetricValue"] { color:#4f8ef7 !important; font-family:'JetBrains Mono',monospace !important; }
#MainMenu, footer { visibility:hidden; }
header { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("models", "bilstm_attention_model.h5")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
MODEL_READY    = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

EXAMPLES = [
    {"title": "Political Propaganda",
     "text": "BREAKING: Scientists CONFIRM that 5G towers are being used by the government to implant tracking microchips in COVID vaccines. Whistleblowers reveal this shocking truth that mainstream media refuses to report. Share before it's deleted!"},
    {"title": "Legitimate News",
     "text": "The Federal Reserve raised its benchmark interest rate by 25 basis points on Wednesday, bringing it to the highest level in 22 years. The move was widely anticipated by markets and reflects the central bank's continued efforts to bring inflation back toward its 2% target."},
    {"title": "Clickbait Misinformation",
     "text": "SHOCKING: This one fruit DESTROYS cancer cells overnight! Big Pharma has been hiding this cure for decades. Doctors HATE this simple trick that cures all diseases. Click here to see what they don't want you to know!"},
]

# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached():
    from predict import _load_artifacts
    return _load_artifacts()

def run_prediction(text):
    from predict import predict
    return predict(text)

def conf_bar(value, kind):
    pct = int(value * 100)
    css = "conf-bar-fake" if kind == "FAKE" else "conf-bar-real"
    return f"""
    <div style="margin:4px 0 12px 0">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8892b0;margin-bottom:4px">
        <span>{kind}</span><span>{pct}%</span>
      </div>
      <div class="conf-bar-bg"><div class="{css}" style="width:{pct}%"></div></div>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FakeScope")
    st.markdown("*AI-powered fake news detector*")
    st.markdown("---")
    st.markdown("### Model Info")
    c1, c2 = st.columns(2)
    c1.metric("Architecture", "BiLSTM")
    c2.metric("Mechanism", "Attention")
    st.metric("Max Tokens", "500")
    st.metric("Vocab Size", "50 K")
    st.markdown("---")
    st.markdown("### Quick Examples")
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"📄 {ex['title']}", key=f"ex_{i}"):
            st.session_state["input_text"] = ex["text"]
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
1. **Preprocessing** — tokenize, lemmatize, remove stopwords
2. **Embedding** — maps tokens to dense vectors
3. **BiLSTM** — captures bidirectional context
4. **Attention** — weights important phrases
5. **Dense classifier** — outputs Fake / Real probability
    """)
    st.caption("Built with TensorFlow · Keras · Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 8px 0">
  <div style="font-size:3rem;margin-bottom:8px">🔍</div>
  <h1 style="font-size:2.4rem;font-weight:800;
     background:linear-gradient(90deg,#4f8ef7,#a78bfa);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
     FakeScope</h1>
  <p style="color:#8892b0;font-size:1.05rem;margin-top:6px">
    Detect fake news using BiLSTM + Attention deep learning</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────────────────────────
if not MODEL_READY:
    st.warning("⚠️ Trained model not found. Push `models/bilstm_attention_model.h5` and `models/tokenizer.pkl` to the repo.", icon="⚠️")
else:
    with st.spinner("Loading model…"):
        try:
            load_model_cached()
            st.success("✅ Model loaded and ready!", icon="✅")
        except Exception as e:
            st.error(f"Model load error: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📝 Enter News Article")
input_text = st.text_area(
    label="news input",
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
    st.metric("Words", len(input_text.split()) if input_text.strip() else 0)
with col_c:
    st.metric("Chars", len(input_text))
st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze_btn:
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    elif len(input_text.split()) < 5:
        st.warning("Please enter at least 5 words for a reliable prediction.")
    else:
        with st.spinner("Analyzing…"):
            t0 = time.time()
            try:
                result  = run_prediction(input_text)
                elapsed = time.time() - t0
                label   = result["label"]
                conf    = result["confidence"]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Analysis Result")

                r1, r2, r3 = st.columns([2, 2, 1])
                with r1:
                    pill = "result-fake" if label == "FAKE" else "result-real"
                    icon = "🚨" if label == "FAKE" else "✅"
                    st.markdown(
                        f'<p style="color:#8892b0;font-size:0.85rem;margin-bottom:6px">VERDICT</p>'
                        f'<span class="{pill}">{icon} {label}</span>',
                        unsafe_allow_html=True)
                with r2:
                    st.metric("Confidence", f"{conf*100:.1f}%")
                with r3:
                    st.metric("Time", f"{elapsed*1000:.0f}ms")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probability breakdown**")
                st.markdown(conf_bar(result["fake_prob"], "FAKE"), unsafe_allow_html=True)
                st.markdown(conf_bar(result["real_prob"], "REAL"), unsafe_allow_html=True)
                st.markdown("---")

                if label == "FAKE":
                    msg = "🚨 **High Risk** — Strong indicators of fake/misleading news. Cross-check with trusted sources." if conf >= 0.75 else "🟡 **Moderate Risk** — Some misleading patterns detected. Verify independently."
                else:
                    msg = "✅ **Likely Credible** — Strong indicators of factual reporting." if conf >= 0.75 else "🟡 **Uncertain** — Cannot determine with high confidence. Verify independently."
                st.info(msg)

                with st.expander("🔬 View preprocessed text"):
                    st.code(result["cleaned_text"], language=None)

                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("⚠️ For educational purposes only. Always verify from authoritative sources.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

# ── Batch ─────────────────────────────────────────────────────────────────────
with st.expander("📂 Batch Analysis (upload CSV)"):
    st.markdown("Upload a CSV with a **`text`** column to analyze multiple articles.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
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
                        results.append({"text": str(row["text"])[:80]+"…",
                                        "label": r["label"],
                                        "confidence": f"{r['confidence']*100:.1f}%"})
                    except Exception:
                        results.append({"text": str(row["text"])[:80], "label": "ERROR", "confidence": "-"})
                    prog.progress((i+1)/len(df_up))
                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True)
                st.download_button("⬇️ Download Results", df_res.to_csv(index=False).encode(),
                                   "results.csv", "text/csv")

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown("""
### Fake News Detection using BiLSTM + Attention
| Component | Details |
|---|---|
| Embedding | Trainable word embeddings (128-dim) |
| Encoder | BiLSTM (64 units each direction) |
| Attention | Bahdanau-style soft attention |
| Classifier | Dense(64) → Dropout → Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam |
    """)
