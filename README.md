# Sentiment Analysis with BiLSTM  

## 📌 Project Overview  
This project implements **Sentiment Analysis** using a **Bidirectional LSTM (BiLSTM)** neural network. The goal is to classify text data into **positive** or **negative sentiment**, leveraging the ability of BiLSTMs to capture contextual information from both past and future tokens in a sequence.  

The pipeline includes:  
- Data preprocessing and train/validation/test splitting.  
- Model training with **early stopping** to prevent overfitting.  
- Performance visualization (accuracy and loss).  
- Evaluation on test data.  

---

## ⚙️ Installation and Setup  

### 1. Clone the repository  
```bash
git clone https://github.com/anniemburu/Sentimental-Analysis-with-BiLSTM
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
The data used from this project was sourced from [Movie Review, Polarity Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz). You can modify data_split in src/data/preprocessing.py if you wish to use a different dataset. You can modify ``` bash data_split ``` in ```bash src/data/preprocessing.py``` if you wish to use a different dataset.

## 🚀 Training the Model

Run the training pipeline with:

```bash
python train.py
```
This will:

- Train the BiLSTM model on the training data.
- Validate it on the validation set.
- Save the trained model to ```bash src/models/model_final.h5 ``` .
- Generate training performance plots at ```bash src/results/model_performance.png ```.

## 📊 Data Source

The data is sourced from [Kaggle](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset/data). You can either download it manually or using Kaggle APIs. Downloading requires you to have a kaggle account if you want to use this data. You can use your own data.

    ## Download Data with Kaggle API

        1. Go to kaggle.com → Your profile → Settings → "Create New API Token"
        2. Download ```bash kaggle.json ```.
        3. In the folder ```bash ~\Users\<YourUsername>\ ```, check if ```bash .kaggle ``` folder exists. If not create one using ```bash mkdir -p ~/.kaggle```
        4. Place json file downloaded in ```bash ~/.kaggle/ on Linux/Mac or C:\Users\<YourUsername>\.kaggle\ ``` on Windows or ```bash ~/.kaggle/ ``` on Linux/ . 
        5. Set permission on Linux/Mac using ```bash chmod 600 ~/.kaggle/kaggle.json ```.

    ## Data SetUp

    - The processed data is stored in: ```bash datasets/processed/sentiment_data.csv```. 
    - Manually downladed data can be stored at  ```bash datasets/raw/. You can modify ``` bash data_split ``` in ```bash src/data/preprocessing.py``` if you wish to use a different dataset.
    - In the file ```bash src/data_loader.py``` file, change the kaggle name and key with your own.
    - To automatically download from kaggle add the arguement ```bash --autodownload ``` or ```bash -autodownload ``` when training.


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

