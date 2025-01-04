import pickle
import gradio as gr
import numpy as np
from gensim.models import Word2Vec

# Load the logistic regression model
with open('models/llogistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.model')

def sentence_vector(tokens, model):
    """Calculate the sentence vector by averaging word vectors."""
    valid_words = [word for word in tokens if word in model.wv]
    if valid_words:
        return np.mean(model.wv[valid_words], axis=0)
    else:
        return np.zeros(model.vector_size)

def classify_comment(comment):
    """Classify the sentiment of a comment as bearish, bullish, or neutral."""
    try:
        # Tokenize the comment
        tokens = comment.lower().split()

        # Generate sentence vector using Word2Vec
        processed_comment = sentence_vector(tokens, word2vec_model).reshape(1, -1)
        
        # Predict sentiment
        prediction = model.predict(processed_comment)[0]
        
        # Map prediction to labels (ensure the model output aligns with these labels)
        sentiment_map = {0: "neutral", 1: "bullish", 2: "bearish"}
        sentiment = sentiment_map.get(prediction, "unknown")

        return sentiment
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=classify_comment,
    inputs=gr.Textbox(label="Enter your comment (e.g., about BTC or stock markets):"),
    outputs=gr.Label(label="Sentiment"),
    title="BTC Sentiment Analyzer",
    description="Predict whether a comment is bullish, bearish, or neutral using a logistic regression model."
)

# Launch the Gradio interface
interface.launch()
