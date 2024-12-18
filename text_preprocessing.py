import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessing:
    def __init__(self, filepath: str):

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        self.data = pd.read_csv(filepath)
        self.processed_data = self.data.copy()

        self.stop_words = set(stopwords.words('english'))

    def clean_tweet(self, tweet: str) -> str:
        tweet = tweet.lower()
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet
    def remove_stopwords(self, tweet: str) -> str:
        words = word_tokenize(tweet)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    def preprocess_tweets(self) -> pd.DataFrame:
        self.processed_data['cleaned_text'] = self.processed_data['text'].apply(self.clean_tweet)
        self.processed_data['processed_text'] = self.processed_data['cleaned_text'].apply(self.remove_stopwords)
        return self.processed_data





