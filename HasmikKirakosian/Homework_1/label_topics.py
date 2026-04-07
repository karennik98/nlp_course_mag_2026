import json
from gensim import models, corpora

# Բեռնում
lda_model = models.LdaModel.load('models/lda_model.model')
dictionary = corpora.Dictionary.load('models/dictionary.dict')

topic_labels = {}

print("--- Թեմաների անվանում ---")
for i in range(10):
    words = lda_model.show_topic(i, topn=20)
    topic_words = ", ".join([w[0] for w in words])
    print(f"\nTopic {i} բառերը: {topic_words}")
    
    name = input(f"Մուտքագրեք անուն Topic {i}-ի համար (կամ Enter): ")
    topic_labels[str(i)] = name if name else f"Topic {i}"

# Պահպանում
with open('models/topic_labels.json', 'w') as f:
    json.dump(topic_labels, f)

print("\n--- Ամփոփում ---")
for id, label in topic_labels.items():
    print(f"Topic {id}: {label}")
