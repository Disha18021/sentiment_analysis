import streamlit as st
from transformers import pipeline
from langchain.chains.base import Chain
from typing import Any, Dict

# Specify the model explicitly
#sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define a simple LLMChain using LangChain
class SentimentChain(Chain):
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text = inputs['text']
        result = sentiment_analysis(text)
        # The result is a list of dictionaries, extract the first result
        result = result[0]
        return {
            "label": result.get('label', 'UNKNOWN'),
            "score": result.get('score', 0.0)
        }

    @property
    def input_keys(self) -> list:
        return ["text"]

    @property
    def output_keys(self) -> list:
        return ["label", "score"]

# Initialize the sentiment analysis chain
sentiment_chain = SentimentChain()

# Emojis for sentiments
sentiment_emojis = {
    "POSITIVE": "😊",
    "NEGATIVE": "😞",
    "NEUTRAL": "😐",
    "UNKNOWN": "🤔"
}

# Streamlit app
st.title("Sentiment Analysis Tool")
st.write("Enter text to analyze its sentiment.")

user_input = st.text_area("Text", "")
if st.button("Analyze Sentiment"):
    if user_input:
        try:
            result = sentiment_chain.invoke({"text": user_input})
            label = result.get('label', 'UNKNOWN')
            score = result.get('score', 0.0)

            # Heuristic for neutral sentiment
            if label == 'POSITIVE' and score < 0.55:
                label = 'NEUTRAL'
            elif label == 'NEGATIVE' and score > 0.45:
                label = 'NEUTRAL'
            emoji = sentiment_emojis.get(label, sentiment_emojis['UNKNOWN'])

            #emoji = sentiment_emojis.get(label, "🤔")

            st.write("Sentiment Analysis Result:")
            st.write(f"Label: {label} {emoji}")
            st.write(f"Score: {score:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter some text to analyze.")
