from load_dataset_from_HF import DatasetDownloader
from text_preprocessing import Preprocessing

def main():
    filepath = '/Users/efecelik/Desktop/investor-sentiments/datasets/zeroshot_twitter-financial-news-sentiment_train.csv'

    preprocessor = Preprocessing(filepath)

    processed_data = preprocessor.preprocess_tweets()
    print(processed_data.head())
