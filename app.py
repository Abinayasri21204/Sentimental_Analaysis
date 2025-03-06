import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model from Hugging Face
sentiment_model = pipeline("sentiment-analysis")

# Define a function for prediction
def analyze_sentiment(text):
    result = sentiment_model(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

# Create a Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter text for Sentiment Analysis"),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analysis with Hugging Face & Gradio",
    description="Enter a sentence and get its sentiment (Positive/Negative)."
)

# Launch the Gradio app
iface.launch()
