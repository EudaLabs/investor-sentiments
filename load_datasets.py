import os
import pandas as pd
from datasets import load_dataset, list_datasets
import huggingface_hub


class DatasetDownloader:
    def __init__(self, save_directory='/Users/efecelik/Desktop/investor-sentiments/datasets'):
        """
        Initialize dataset downloader

        Parameters:
        - save_directory: Directory to save downloaded datasets
        """
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)

    def list_recommended_finance_datasets(self):
        """
        List recommended finance and sentiment datasets

        Returns:
        - List of recommended datasets
        """
        recommended_datasets = [
            # Verified working datasets
            # 'takala/financial_phrase_bank',
            # 'lachmannflo/stock_headlines_sentiment_dataset',
            # 'marketintelligence/sentiment_financial_headlines',
            # 'enron_sentiment',
            'zeroshot/twitter-financial-news-sentiment'
        ]

        return recommended_datasets

    def download_dataset(self, dataset_name, split='train'):
        """
        Download a specific dataset

        Parameters:
        - dataset_name: Name of the dataset to download
        - split: Dataset split to download (default: 'train')

        Returns:
        - Pandas DataFrame of the dataset
        """
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split=split)

            # Convert to pandas DataFrame
            df = dataset.to_pandas()

            # Create filename
            filename = f"{dataset_name.replace('/', '_')}_{split}.csv"
            filepath = os.path.join(self.save_directory, filename)

            # Save to CSV
            df.to_csv(filepath, index=False)

            print(f"Dataset {dataset_name} downloaded and saved to {filepath}")
            print(f"Dataset shape: {df.shape}")
            print("\nColumns:")
            print(df.columns)

            return df

        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None

    def bulk_download(self, datasets=None):
        """
        Download multiple datasets

        Parameters:
        - datasets: List of dataset names (if None, use recommended list)

        Returns:
        - Dictionary of downloaded datasets
        """
        if datasets is None:
            datasets = self.list_recommended_finance_datasets()

        downloaded_datasets = {}

        for dataset_name in datasets:
            try:
                df = self.download_dataset(dataset_name)
                if df is not None:
                    downloaded_datasets[dataset_name] = df
            except Exception as e:
                print(f"Failed to download {dataset_name}: {e}")

        return downloaded_datasets


def main():
    # Initialize downloader
    downloader = DatasetDownloader()

    # Print recommended datasets
    print("Recommended Finance Datasets:")
    recommended = downloader.list_recommended_finance_datasets()
    for dataset in recommended:
        print(f"- {dataset}")

    # Option to download
    download_choice = input("\nDo you want to download these datasets? (yes/no): ").lower()

    if download_choice in ['yes', 'y']:
        # Bulk download
        downloaded = downloader.bulk_download()

        print("\nDownload Summary:")
        for name, df in downloaded.items():
            print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()