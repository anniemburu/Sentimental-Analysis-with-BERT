import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import yaml

from src.data.data_loader import data_load
from src.data.preprocessing import SentimentalData

class TrainingPipeline():
    def __init__(self, model, optimizer, device):
        self.model = model
        #self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_model(self, train_loader):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0 

        for batch in train_loader:
            self.optimizer.zero_grad()

            #add data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            #output (forward pass)
            output = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels  
            )

            loss = output.loss
            total_loss += loss.item()

            predictions = torch.argmax(output.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loss.backward() #backward pass
            self.optimizer.step() #update model params

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        # Store in history
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)

        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                output = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels  
                )

                # Calculate loss
                loss = output.loss
                total_loss += loss.item()

                predictions = torch.argmax(output.logits, dim = 1)
                correct += (predictions == labels).sum().item() #get correctly predicted
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        # Store in history
        self.history['val_loss'].append(avg_loss)
        self.history['val_accuracy'].append(accuracy)

        return avg_loss, accuracy
    
    
    def output_model(self):
        return self.model
    

# Main Scripts
if __name__ == "__main__":

    print("Let the games begin ....")

    data_load() #download data or not...

    #download data
    data_old = pd.read_csv('datasets/processed/sentiment_data.csv')

    data = data_old.copy()

    label_encoder = LabelEncoder()
    data['Sentiment'] = label_encoder.fit_transform(data_old['Sentiment'])

    train_text, val_text, train_label, val_label = train_test_split(data['Text'], data['Sentiment'], train_size=0.8, random_state=42, )

    class_names = [str(cls) for cls in label_encoder.classes_] #classes
    label_mapping = dict(zip(range(len(class_names)), class_names))  #map of class and int value

    #
    with open('config/vars.yml', "w") as f:
        yaml.dump({"label_mapping": label_mapping}, f)
        
    #params for model
    model_name = 'bert-base-uncased'
    max_length = 128
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device : {device}")

    #init model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels = len(class_names)).to(device)
    
    #Create datasets 
    train_dataset = SentimentalData(train_text, train_label, tokenizer, max_length)
    val_dataset = SentimentalData(val_text, val_label, tokenizer, max_length)

    #Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    #Init Optimizer
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    #training pipeline
    training_pipeline = TrainingPipeline(model, optimizer, device)

    ##Training
    print("\n Starting Training...")

    for epoch in range(epochs):
        train_loss, train_accuracy = training_pipeline.train_model(train_loader)
        val_loss, val_accuracy = training_pipeline.evaluate(val_loader)

        print(f"Epoch {epoch+1}/{epochs},   Train Loss: {train_loss:.4f},   Train Accuracy: {train_accuracy:.4f} , Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    final_model = training_pipeline.output_model() #get final modle

    #save the models for future use....
    final_model.save_pretrained('src/models/sentiment_model')
    tokenizer.save_pretrained('src/models/sentiment_model')
    print("Model saved to 'src/models/sentiment_model'") 


    print("End of Training .....")




