from draft_code.text_preprocessing import Preprocessing

def main():
    filepath = '/datasets/zeroshot_twitter-financial-news-sentiment_train.csv'

    preprocessor = Preprocessing(filepath)

    processed_data = preprocessor.preprocess_tweets()
    print(processed_data.head())
