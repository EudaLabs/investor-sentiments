import pickle
import numpy as np
from gensim.models import Word2Vec
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# MODEL LOADING WITH ERROR HANDLING
# ---------------------------

def load_models(model_paths):
    """
    Load all required models with error handling
    Returns a dictionary of loaded models
    """
    models = {}
    try:
        # Logistic Regression Model
        with open(model_paths['logistic'], 'rb') as file:
            models['logistic_model'] = pickle.load(file)
        logger.info("Logistic Regression model loaded successfully")

        # Neural Network Model
        models['nn_model'] = load_model(model_paths['nn'])
        logger.info("Neural Network model loaded successfully")

        # LSTM Model
        models['lstm_model'] = load_model(model_paths['lstm'])
        logger.info("LSTM model loaded successfully")

        # Word2Vec Model
        models['word2vec_model'] = Word2Vec.load(model_paths['word2vec'])
        logger.info("Word2Vec model loaded successfully")

        return models
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


# Model paths
MODEL_PATHS = {
    'logistic': '/Users/efecelik/Desktop/investor-sentiments/llogistic_model.pkl',
    'nn': '/Users/efecelik/Desktop/investor-sentiments/sentiment_nn_model.h5',
    'lstm': '/Users/efecelik/Desktop/investor-sentiments/lstm_model.h5',
    'word2vec': '/content/drive/MyDrive/NLP/word2vec_model.model'
}

# Load models
try:
    MODELS = load_models(MODEL_PATHS)
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    MODELS = None


# ---------------------------
# SENTENCE VECTORIZATION
# ---------------------------

def sentence_vector(tokens, model):
    """Calculate the sentence vector by averaging word vectors."""
    try:
        valid_words = [word for word in tokens if word in model.wv]
        if valid_words:
            return np.mean(model.wv[valid_words], axis=0)
        return np.zeros(model.vector_size)
    except Exception as e:
        logger.error(f"Error in sentence vectorization: {str(e)}")
        raise


# ---------------------------
# COMMENT CLASSIFICATION
# ---------------------------

def classify_comment(comment, selected_model):
    """Classify the sentiment of a comment using the selected model."""
    if MODELS is None:
        return "Error: Models not properly loaded"

    try:
        # Input validation
        if not comment or not isinstance(comment, str):
            return "Error: Invalid input comment"

        # Tokenize the comment
        tokens = comment.lower().split()

        # Generate sentence vector
        processed_comment = sentence_vector(tokens, MODELS['word2vec_model']).reshape(1, -1)

        # Model Selection and Prediction
        sentiment_map = {0: "neutral", 1: "bullish", 2: "bearish"}

        if selected_model == "Logistic Regression":
            prediction = MODELS['logistic_model'].predict(processed_comment)[0]
        elif selected_model == "Neural Network":
            prediction = np.argmax(MODELS['nn_model'].predict(processed_comment), axis=1)[0]
        elif selected_model == "LSTM":
            prediction = np.argmax(MODELS['lstm_model'].predict(processed_comment), axis=1)[0]
        else:
            return "Error: Invalid model selected"

        sentiment = sentiment_map.get(prediction, "unknown")
        logger.info(f"Prediction made successfully: {sentiment}")
        return sentiment

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        return f"Error: {str(e)}"


# ---------------------------
# GRADIO INTERFACE
# ---------------------------

interface = gr.Interface(
    fn=classify_comment,
    inputs=[
        gr.Textbox(label="Enter your comment (e.g., about BTC or stock markets):"),
        gr.Dropdown(
            label="Select Model",
            choices=["Logistic Regression", "Neural Network", "LSTM"],
            value="Logistic Regression"
        )
    ],
    outputs=gr.Label(label="Sentiment"),
    title="BTC Sentiment Analyzer",
    description="Predict whether a comment is bullish, bearish, or neutral using a selected model."
)

# ---------------------------
# APPLICATION LAUNCH
# ---------------------------

if __name__ == "__main__":
    # Configure GPU memory growth if available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth configured")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {str(e)}")

    # Launch the interface
    interface.launch()