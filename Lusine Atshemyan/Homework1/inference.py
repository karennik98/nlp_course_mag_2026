import json
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

lda_model = LdaModel.load("models/lda_model.model")
dictionary = corpora.Dictionary.load("models/dictionary.dict")

with open("models/topic_labels.json", "r") as f:
    topic_labels = json.load(f)

def preprocess(text):
    tokens = simple_preprocess(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    return tokens

def classify_document(text):

    tokens = preprocess(text)

    bow = dictionary.doc2bow(tokens)

    topics = lda_model.get_document_topics(bow)

    topics = sorted(topics, key=lambda x: x[1], reverse=True)

    print("\nTop 3 topics:\n")

    for topic_id, prob in topics[:3]:
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")

        print(f"Topic: {topic_name}")
        print(f"Probability: {prob:.4f}")

        words = lda_model.show_topic(topic_id, topn=5)
        word_list = ", ".join([w for w, _ in words])

        print(f"Top Words: {word_list}\n")

def show_all_topics():

    for topic_id in range(lda_model.num_topics):
        topic_name = topic_labels.get(str(topic_id), f"Topic {topic_id}")
        words = lda_model.show_topic(topic_id, topn=5)
        word_list = ", ".join([w for w, _ in words])

        print(f"{topic_id}: {topic_name}")
        print(f"Top words: {word_list}")


sample_docs = [
    "The new graphics card delivers amazing performance for gaming. The GPU can handle 4K resolution easily with ray tracing enabled. Gamers will love the improved frame rates.",

    "Scientists discovered a new exoplanet orbiting a distant star in the habitable zone. The research team published their findings in Nature journal. This discovery could provide insights into planetary formation.",

    "The basketball team won the championship after an incredible final game. The players celebrated with fans in the stadium. It was the team's first title in twenty years.",

    "Congress passed a new bill regarding healthcare reform. The president is expected to sign the legislation next week. The policy will affect millions of citizens across the country.",

    "I love cooking Italian food at home. Pasta carbonara and margherita pizza are my favorite dishes to make. Fresh ingredients make all the difference in authentic recipes."
]

show_all_topics()

for i, doc in enumerate(sample_docs, 1):
    print(f"\n\nSample {i}")
    print(doc)
    classify_document(doc)