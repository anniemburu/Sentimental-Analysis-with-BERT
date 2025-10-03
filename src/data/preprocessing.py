import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import re
import html
import contractions
import emoji

class SentimentalData(Dataset):
    def __init__(self, texts, sentiments, tokenizer, max_length):
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        sentiment = self.sentiments.iloc[idx]

        preproc = PreProcessing(text)
        text = preproc.preprocessing()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'label' : torch.tensor(sentiment, dtype=torch.long)
        }

class PreProcessing():
    def __init__(self,text):
        self.text = text

    def normalize_text(self, uncased=True):
        self.text = self.text.strip()
        if uncased:
            self.text = self.text.lower()
        
    
    def clean_text(self):
        self.text = contractions.fix(self.text)
        self.text = html.unescape(self.text)  # decode HTML entities
        self.text = re.sub(r"http\S+|www.\S+", "[URL]", self.text)  # replace URLs
        self.text = re.sub(r"@\w+", "[USER]", self.text)           # replace mentions
        self.text = re.sub(r"#(\w+)", r"\1", self.text)            # hashtags: #love -> love
        self.text = re.sub(r"\s+", " ", self.text).strip()         # collapse whitespace
        self.text = re.sub(r"<.*?>", "", self.text)
        self.text = re.sub(r"[^a-zA-Z\s]", "", self.text)
        
    def handle_emojis(self):
        return emoji.demojize(self.text, delimiters=(" ", " "))
    
    def preprocessing(self):

        self.normalize_text()
        self.clean_text()
        self.handle_emojis()

        return self.text
    
        




