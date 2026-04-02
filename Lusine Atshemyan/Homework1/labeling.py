from gensim.models import LdaModel
from gensim.corpora import Dictionary
import json

lda_model = LdaModel.load("models/lda_model.model")
data_dict = Dictionary.load("models/dictionary.dict")

# print(dictionary)

topics = lda_model.show_topics(
    num_topics=-1,
    num_words=20,
    formatted=False
)

# print(topics)
print(f"{'Topic ID':<10} {'Word':<15} {'Probability':<12}")
print("-" * 40)

for topic_id, words in topics:
    for word, prob in words:
        print(f"{topic_id:<10} {word:<15} {prob:<12.4f}")

topic_labels = {}

for topic_id, words in topics:
    print("-" * 40)
    print(f"\nTopic {topic_id}")
    
    for word, prob in words:
        print(f"{word} ({prob:.4f})")
    
    name = input("\nEnter a meaningful name for this topic: ")
    
    if name.strip() == "":
        topic_labels[topic_id] = f"Topic {topic_id}"
    else:
        topic_labels[topic_id] = name.strip()

with open("models/topic_labels.json", "w") as f:
    json.dump(topic_labels, f, indent=4)

print("-" * 50)
print("\nSummary")

for topic_id, words in topics:
    print(f"\n{topic_labels[topic_id]} (Topic {topic_id})")
    
    word_list = ", ".join([word for word, _ in words])
    print(word_list)

