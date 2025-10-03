import pandas as pd
import os
import requests
import io
import tarfile
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.parser import get_parser
import pandas as pd

#Data is sourced from Kaggle : 

def data_download(file_path):
    dataset = 'kashishparmar02/social-media-sentiments-analysis-dataset' 

    os.environ['KAGGLE_USERNAME'] = "annemburu" #replace with your Kaggle username
    os.environ['KAGGLE_KEY'] = "5af16e537041c19100e826b1ea7f28bc" # replace with your Kaggle API key

    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_files(dataset, path=file_path, unzip=True)

    print("Download complete!")

def data_load():
    raw_path = 'datasets/raw' #where to save the raw data

    args = get_parser().parse_args()

    if args.autodownload:
        print("Downloading automatically...")
        data_download(raw_path)

    else:
        print("No need to download... Load directly.")

    dataset = pd.read_csv("datasets/raw/sentimentdataset.csv")

    new_dataset = dataset[['Text','Sentiment']]

    new_dataset.to_csv('datasets/processed/sentiment_data.csv')

            