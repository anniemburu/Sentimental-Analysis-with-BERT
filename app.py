from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import yaml

from src.data.preprocessing import PreProcessing

# Load configuration and model at module level
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('src/models/sentiment_model')
tokenizer = BertTokenizer.from_pretrained('src/models/sentiment_model')
MAX_LENGTH = 128

model.to(device)
model.eval()

with open('config/vars.yml', "r") as f:
    config = yaml.safe_load(f)

label_mapping = config["label_mapping"]

app = FastAPI(title="Sentiment Analysis API")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # Preprocess
    preproc = PreProcessing(request.text)  # Fixed: use request.text
    processed_text = preproc.preprocessing()

    # Encode
    encoding = tokenizer.encode_plus(
        processed_text,  # Use preprocessed text
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        probs = torch.softmax(output.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    sentiment = label_mapping[prediction]

    return {
        "text": request.text,
        "sentiment": sentiment,
        "confidence": confidence
    }