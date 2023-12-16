import sys
import json

import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = ['Нейтральный', 'Позитивный', 'Негативный', 'Восторг', 'Страх', 'Злость', 'Отвращение']
tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')

def predict_emotion(text: str) -> str:
    """
        We take the input text, tokenize it, pass it through the model, and then return the predicted label
        :param text: The text to be classified
        :type text: str
        :return: The predicted emotion
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
        
    return LABELS[predicted[0]]

if name == "main":
    text = sys.argv[1:]
    results = predict_emotion(text)
    print(json.dumps(results))