import sys
import json
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

def analyze_sentiment(messages):
    # Используем регулярные выражения для токенизации текста
    tokenizer = RegexTokenizer()
    
    # Загружаем модель для анализа эмоционального окраса
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    
    # Проводим анализ сентимента для каждого сообщения
    results = model.predict(messages, k=5)

    # Форматируем результаты для удобства восприятия
    formatted_results = []
    for message, sentiment in zip(messages, results):
        formatted_result = {
            'positive': sentiment.get('positive', -2),
            'neutral': sentiment.get('neutral', -2),
            'negative': sentiment.get('negative', -2),
            'speech': sentiment.get('speech', -2),
            'skip': sentiment.get('skip', -2),
        }
        formatted_results.append(formatted_result)

    return formatted_results

if __name__ == "__main__":
    # Получаем тексты из аргументов командной строки
    input_texts = sys.argv[1:]
    
    # Запускаем функцию анализа сентимента
    results = analyze_sentiment(input_texts)
    
    # Выводим результат в формате JSON
    print(json.dumps(results))
