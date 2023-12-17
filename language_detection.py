from langdetect import detect
from langcodes import Language
import sys
import json

def detect_language(messages):
    # Используем библиотеку langdetect для определения языка текста
    # и langcodes для получения названия языка на русском
    results = []
    for message in messages:
        language_code = detect(message)
        language_name = Language(language_code).language_name('ru')  # Выбираем 'ru' для русских названий
        capitalized_language = language_name.capitalize()
        language_with_suffix = f"{capitalized_language} язык"
        results.append(language_with_suffix)
    return results

if __name__ == "__main__":
    # Получаем тексты из аргументов командной строки
    texts = sys.argv[1:]
    
    # Запускаем функцию определения языка
    results = detect_language(texts)
    
    # Выводим результат в формате JSON, отключая экранирование ASCII-символов
    print(json.dumps(results, ensure_ascii=False))
