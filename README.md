# Sentiment Analysis with BERT 

## 📌 Project Overview  
This project implements Sentiment Analysis using BERT (Bidirectional Encoder Representations from Transformers). The goal is to classify text data into positive or negative sentiment, leveraging BERT's powerful ability to understand deep contextual relations in text.

The pipeline includes:

- Data loading and preprocessing with a BERT-specific tokenizer.

- Splitting data into training and validation sets.

- Fine-tuning a pre-trained BERT model with early stopping to prevent overfitting.

- Visualizing performance metrics (accuracy and loss).

- Evaluate by testing your own text. 

---

## ⚙️ Installation and Setup  

### 1. Clone the repository  
```bash
git clone https://github.com/anniemburu/Sentimental-Analysis-with-BERT
```

### 2. Create and activate a virtual environment (Recommend Anaconda or miniconda)
```bash
conda create -n myenv python=3.9

conda activate myenv
```

### 3. Install dependencies

All dependencies are listed in requirements.txt. Install them with:

```bash
pip install -r requirements.txt
```

### 4. Data setup

The processed dataset is expected at: 
```bash datasets/processed/sentiment_data.csv```. 
The data used from this project was sourced from [Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset/data). You can modify data_split in src/data/preprocessing.py if you wish to use a different dataset. You can modify ``` bash data_split ``` in ```bash src/data/preprocessing.py``` if you wish to use a different dataset.

## 🚀 Training the Model

Run the training pipeline with:

```bash
python3 train.py
```
or 

```bash
python3 -m train
```

This will:

- Train the BiLSTM model on the training data.
- Validate it on the validation set.
- Save the trained model to ```bash src/models/sentimental_model/ ``` .
- Generate training performance plots at ```bash src/results/model_performance.png ```.


You can add extra parameters as defined in ```bash src/utils/parser.py```.

## 🚀 Evaluate the Model
You can test your own text by running : 

```bash
python3 evaluate.py
```
or 

```bash
python3 -m evaluate
```

## 📊 Data Source

The data is sourced from [Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset/data). You can either download it manually or automatically. 

```bash
python3 -m train --autodownload
```


Preprocessing: The data has been tokenized, padded to fixed sequence length, and split into training and testing.

## 📂 Project Structure

```bash
├── config
│   └── vars.yml
├── datasets/
│   ├── processed/
│   │   └── sentiment_data.csv
│   └── raw/
│       └── sentimentdataset.csv
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── sentimental_model/
│   ├── results/
│   │   └── model_performance.png
│   └── utils/
│       └── parser.py
├── evaluate.py
├── train.py                
├── requirements.txt
└── README.md

```

## 🔎 Findings & Results
TBA

