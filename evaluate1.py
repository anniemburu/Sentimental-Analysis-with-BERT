from src.data.preprocessing import data_split
import pandas as pd
from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def pre_evaluation(model_path, test_data):

    X_test, y_test = test_data['text'], test_data['sentiment']

    model = load_model(model_path)

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")


    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


def custom_evaluation(model_path, test_data):
    pass

if __name__ == "__main__":
    data = pd.read_csv('datasets/processed/sentiment_data.csv')

    train_data, val_data, test_data = data_split(data)

    model_path = "src/models/model_final.h5"

    pre_evaluation(model_path, test_data)