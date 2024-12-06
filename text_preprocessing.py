import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy


class TextPreprocessor:
    def __init__(self, load_path=None):
        """
        Initialize the TextPreprocessor with optional dataset loading

        Args:
            load_path (str, optional): Path to the CSV file to load
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        # Load spaCy English model
        self.nlp = spacy.load('en_core_web_sm')

        # Load dataset if path is provided
        self.df = pd.read_csv(load_path) if load_path else None

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Set up stop words
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Perform comprehensive text cleaning

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text):
        """
        Remove stopwords from text

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def lemmatize_text(self, text):
        """
        Lemmatize words in the text

        Args:
            text (str): Input text

        Returns:
            str: Lemmatized text
        """
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def named_entity_extraction(self, text):
        """
        Extract named entities from text

        Args:
            text (str): Input text

        Returns:
            list: Named entities found in the text
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def preprocess_pipeline(self, text):
        """
        Complete text preprocessing pipeline

        Args:
            text (str): Input text

        Returns:
            str: Fully preprocessed text
        """
        cleaned_text = self.clean_text(text)
        no_stopwords_text = self.remove_stopwords(cleaned_text)
        lemmatized_text = self.lemmatize_text(no_stopwords_text)

        return lemmatized_text

    def process_dataset(self):
        """
        Apply preprocessing to entire dataset

        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Provide a file path during initialization.")

        # Create a copy to avoid modifying original data
        processed_df = self.df.copy()

        # Apply preprocessing pipeline to text column
        processed_df['processed_text'] = processed_df['text'].apply(self.preprocess_pipeline)

        return processed_df

    def get_text_statistics(self):
        """
        Compute basic text statistics

        Returns:
            dict: Text-related statistics
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Provide a file path during initialization.")

        return {
            'total_texts': len(self.df),
            'avg_text_length': self.df['text'].str.len().mean(),
            'label_distribution': self.df['label'].value_counts(normalize=True)
        }


# Example usage
def main():
    # Specify your file path
    file_path = '/Users/efecelik/Desktop/investor-sentiments/datasets/zeroshot_twitter-financial-news-sentiment_train.csv'

    # Initialize preprocessor
    preprocessor = TextPreprocessor(load_path=file_path)

    # Get text statistics
    stats = preprocessor.get_text_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Process dataset
    processed_df = preprocessor.process_dataset()

    # Optional: Save processed dataset
    processed_df.to_csv('processed_financial_news.csv', index=False)

    # Demonstrate named entity extraction on a sample text
    sample_text = processed_df['text'].iloc[0]
    entities = preprocessor.named_entity_extraction(sample_text)
    print("\nNamed Entities:")
    for entity, label in entities:
        print(f"{entity} ({label})")


if __name__ == "__main__":
    main()