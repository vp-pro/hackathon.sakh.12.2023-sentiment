import sys
import json
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

# Задаем метки для эмоциональных окрасов
LABELS = ['Нейтральный эмоциональный окрас', 'Позитивный эмоциональный окрас', 'Негативный эмоциональный окрас', 'Восторг', 'Страх', 'Злость', 'Отвращение']

# Загружаем токенизатор и модель для классификации эмоционального окраса
tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')

def predict_emotion(texts: list) -> list:
    """
    Принимает список текстов, токенизирует каждый, передает через модель
    и возвращает список предсказанных эмоциональных окрасов.
    
    :param texts: Список текстов для классификации
    :type texts: list of str
    :return: Список предсказанных эмоциональных окрасов
    """
    # Токенизируем тексты и создаем пакет для передачи в модель
    inputs = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    # Пропускаем тексты через модель
    outputs = model(**inputs)
    
    # Применяем функцию Softmax и выбираем индекс с максимальной вероятностью
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    
    # Возвращаем список предсказанных эмоциональных окрасов с учетом меток
    return [LABELS[label] for label in predicted]

if __name__ == "__main__":
    # Получаем тексты из аргументов командной строки
    texts = sys.argv[1:]
    
    # Запускаем функцию предсказания эмоциональных окрасов
    results = predict_emotion(texts)
    
    # Выводим результат в формате JSON
    print(json.dumps(results))
