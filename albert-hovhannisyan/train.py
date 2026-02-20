import os
from sklearn.datasets import fetch_20newsgroups
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

NUM_TOPICS = 10
NUM_PASSES = 15
TOP_N_WORDS = 15
MODELS_DIR = "models"

print("Loading 20 Newsgroups dataset (first 1000 documents)...")
newsgroups = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
)
documents = newsgroups.data[:1000]
print(f"Loaded {len(documents)} documents.")

def preprocess(text):
    """Tokenize, lowercase, remove stopwords, keep words >= 3 chars."""
    tokens = []
    for word in text.lower().split():
        word = "".join(ch for ch in word if ch.isalpha())
        if len(word) >= 3 and word not in STOPWORDS:
            tokens.append(word)        
    return tokens


print("Preprocessing documents...")
tokenized_docs = [preprocess(doc) for doc in documents]

print("Building dictionary and corpus...")
dictionary = corpora.Dictionary(tokenized_docs)

dictionary.filter_extremes(no_below=5, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
print(f"Dictionary size after filtering: {len(dictionary)} unique tokens")
print(f"Corpus size: {len(corpus)} documents")

print(f"\nTraining LDA model with {NUM_TOPICS} topics, {NUM_PASSES} passes...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=NUM_PASSES,
    alpha="auto",
    eta="auto",
    random_state=42,
)
print("Training complete.")

os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "lda_model")
dict_path = os.path.join(MODELS_DIR, "dictionary")

lda_model.save(model_path)
dictionary.save(dict_path)
print(f"\nModel saved to '{model_path}'")
print(f"Dictionary saved to '{dict_path}'")

print(f"\n{'='*60}")
print(f"DISCOVERED TOPICS (top {TOP_N_WORDS} words each)")
print(f"{'='*60}")
for topic_id in range(NUM_TOPICS):
    words = lda_model.show_topic(topic_id, topn=TOP_N_WORDS)
    word_list = ", ".join(f"{w}" for w, _ in words)
    print(f"\nTopic {topic_id:2d}: {word_list}")
print(f"\n{'='*60}")
print("Done. Run 'label_topics.py' to assign meaningful names to topics.")
