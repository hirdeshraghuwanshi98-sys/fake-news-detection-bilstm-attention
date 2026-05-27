import streamlit as st
from predict import predict_news

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
