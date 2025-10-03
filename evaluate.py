
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import yaml

from src.data.preprocessing import PreProcessing

def predict_sentiment(text, tokenizer, model, device, label_mapping, max_length = 128):
    model.eval()

    preproc = PreProcessing(text)
    text = preproc.preprocessing()

    #encode
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_length,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
        probs = torch.softmax(output.logits, dim = 1)

        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    sentiment = label_mapping[prediction] 

    return sentiment, confidence

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model = BertForSequenceClassification.from_pretrained('src/models/sentiment_model')
    tokenizer = BertTokenizer.from_pretrained('src/models/sentiment_model')


    #load mapping
    with open('config/vars.yml', "r") as f:
        config = yaml.safe_load(f)

    label_mapping = config["label_mapping"]

    print("\n" + "="*50)
    print("Interactive Sentiment Analysis")
    print("Type your text or 'quit' to exit")
    print("="*50 + "\n")

    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue

        sentiment, confidence = predict_sentiment(
            text, tokenizer, model, device, label_mapping, max_length = 128
            )

        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})\n")