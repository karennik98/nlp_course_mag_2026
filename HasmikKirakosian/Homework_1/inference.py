import json
from gensim import models, corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# 1. Բեռնել մոդելը, բառարանը և պիտակները [cite: 107, 111]
lda_model = models.LdaModel.load('models/lda_model.model')
dictionary = corpora.Dictionary.load('models/dictionary.dict')
with open('models/topic_labels.json', 'r') as f:
    topic_labels = json.load(f)

# Ցուցադրել բեռնված թեմաների ամփոփումը [cite: 127]
print("--- Բեռնված թեմաներ ---")
for id, label in topic_labels.items():
    print(f"ID {id}: {label}")

# 2. Դասակարգման ֆունկցիա [cite: 115]
def classify_document(text):
    # Նախամշակում [cite: 117]
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS and len(t) >= 3]
    # Bag-of-Words ձևաչափ [cite: 118]
    bow = dictionary.doc2bow(tokens)
    
    # Ստանալ թեմաների բաշխումը և վերցնել թոփ 3-ը [cite: 119, 120]
    doc_topics = sorted(lda_model.get_document_topics(bow), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n--- Document Preview ---")
    print(f"{text[:150]}...") # [cite: 122]
    
    for topic_id, prob in doc_topics:
        name = topic_labels[str(topic_id)] # [cite: 123]
        # Վերցնել թեմայի թոփ 5 բառերը [cite: 125]
        top_words = [w[0] for w in lda_model.show_topic(topic_id, 5)]
        print(f"Topic: {name} | Probability: {prob:.2%} [cite: 124]")
        print(f"   Top words: {', '.join(top_words)}")

# 3. Ստուգողական նմուշներ [cite: 128, 129]
samples = [
    "The new graphics card delivers amazing performance for gaming. The GPU can handle 4K resolution easily with ray tracing enabled. Gamers will love the improved frame rates.", # [cite: 132]
    "Scientists discovered a new exoplanet orbiting a distant star in the habitable zone. The research team published their findings in Nature journal. This discovery could provide insights into planetary formation.", # [cite: 134]
    "The basketball team won the championship after an incredible final game. The players celebrated with fans in the stadium. It was the team's first title in twenty years.", # [cite: 136]
    "Congress passed a new bill regarding healthcare reform. The president is expected to sign the legislation next week. The policy will affect millions of citizens across the country.", # [cite: 138]
    "I love cooking Italian food at home. Pasta carbonara and margherita pizza are my favorite dishes to make. Fresh ingredients make all the difference in authentic recipes." # [cite: 140]
]

print("\n--- Դասակարգման արդյունքներ ---")
for s in samples:
    classify_document(s)
