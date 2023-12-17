import sys
import json
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

# Создаем объекты для работы с текстом
segmenter = Segmenter()
morph_vocab = MorphVocab()

# Загружаем эмбеддинги для анализа новостных текстов
emb = NewsEmbedding()

# Инициализируем морфологический таггер, синтаксический парсер и NER-таггер
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

# Инициализируем экстракторы для имен, дат, сумм денег и адресов
names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)

def person_catcher(text):
    # Создаем объект Doc для обработки текста
    doc = Doc(text)
    
    # Сегментируем текст
    doc.segment(segmenter)
    
    # Применяем морфологический анализ
    doc.tag_morph(morph_tagger)
    
    # Применяем синтаксический анализ
    doc.parse_syntax(syntax_parser)
    
    # Применяем NER-анализ
    doc.tag_ner(ner_tagger)
    
    # Создаем список для хранения найденных имен
    persons = []

    # Итерируем по спанам (частям текста), найденным NER-таггером
    for span in doc.spans:
        # Проверяем, что тип спана соответствует типу PER (имя)
        if span.type == PER:
            # Добавляем нормализованное имя в список
            persons.append(span.normal)
    
    # Возвращаем список найденных имен
    return persons

if __name__ == "__main__":
    # Получаем текст из аргументов командной строки
    text = sys.argv[1:]
    
    # Запускаем функцию поиска имен
    results = person_catcher(text)
    
    # Выводим результат в формате JSON
    print(json.dumps(results))
