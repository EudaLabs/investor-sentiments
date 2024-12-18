import os
import pandas as pd
from datasets import load_dataset


class DatasetDownloader:
    def __init__(self, save_dir='/Users/efecelik/Desktop/investor-sentiments/datasets'):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def download(self, dataset_name, split='train'):
        try:
            #Download the dataset
            df = load_dataset(dataset_name, split=split).to_pandas()

            filename = f"{dataset_name.replace('/', '_')}_{split}.csv"
            filepath = os.path.join(self.save_dir, filename)

            df.to_csv(filepath, index=False)

            print(f"Dataset {dataset_name} indirildi: {filepath}")
            print(f"Boyut: {df.shape}")

            return df

        except Exception as e:
            print(f"İndirme hatası {dataset_name}: {e}")
            return None




