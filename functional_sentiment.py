import sys
import json
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

def analyze_sentiment(messages):
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    
    results = model.predict(messages, k=5)

    formatted_results = []
    for message, sentiment in zip(messages, results):
        formatted_result = {
            'positive': sentiment.get('positive', -2),
            'neutral': sentiment.get('neutral', -2),
            'negative': sentiment.get('negative', -2),
            'speech': sentiment.get('speech', -2),
            'skip': sentiment.get('skip', -2),
        }
        formatted_results.append({"message": message, "sentiment": formatted_result})

    return formatted_results

if __name__ == "__main__":
    input_texts = sys.argv[1:]
    results = analyze_sentiment(input_texts)
    print(json.dumps(results))
