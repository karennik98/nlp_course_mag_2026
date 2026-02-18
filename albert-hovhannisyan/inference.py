import json
import os
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

MODELS_DIR = "models"
LABELS_FILE = os.path.join(MODELS_DIR, "topic_labels.json")
TOP_TOPICS = 3
TOP_WORDS_PER_TOPIC = 5
PREVIEW_LENGTH = 120

SAMPLE_DOCUMENTS = [
    (
        "Sample 1 – Technology / Gaming",
        "The new graphics card delivers amazing performance for gaming. "
        "The GPU can handle 4K resolution easily with ray tracing enabled. "
        "Gamers will love the improved frame rates.",
    ),
    (
        "Sample 2 – Science / Space",
        "Scientists discovered a new exoplanet orbiting a distant star in the "
        "habitable zone. The research team published their findings in Nature "
        "journal. This discovery could provide insights into planetary formation.",
    ),
    (
        "Sample 3 – Sports",
        "The basketball team won the championship after an incredible final game. "
        "The players celebrated with fans in the stadium. It was the team's first "
        "title in twenty years.",
    ),
    (
        "Sample 4 – Politics",
        "Congress passed a new bill regarding healthcare reform. The president is "
        "expected to sign the legislation next week. The policy will affect millions "
        "of citizens across the country.",
    ),
    (
        "Sample 5 – Food / Cooking",
        "I love cooking Italian food at home. Pasta carbonara and margherita pizza "
        "are my favorite dishes to make. Fresh ingredients make all the difference "
        "in authentic recipes.",
    ),
]

model_path = os.path.join(MODELS_DIR, "lda_model")
dict_path = os.path.join(MODELS_DIR, "dictionary")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found at '{model_path}'. Run 'train.py' first."
    )

print("Loading model and dictionary...")
lda_model = LdaModel.load(model_path)
dictionary = corpora.Dictionary.load(dict_path)
num_topics = lda_model.num_topics

if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        topic_labels = json.load(f)
    print(f"Topic labels loaded from '{LABELS_FILE}'")
else:
    topic_labels = {str(i): f"Topic_{i}" for i in range(num_topics)}
    print("No label file found – using default topic names.")

print(f"Model ready ({num_topics} topics).\n")

def preprocess(text):
    """Same preprocessing pipeline used during training."""
    tokens = []
    for word in text.lower().split():
        word = "".join(ch for ch in word if ch.isalpha())
        if len(word) >= 3 and word not in STOPWORDS:
            tokens.append(word)
    return tokens


def classify(text):
    """
    Return the top-N topic distributions for the given text document.
    Each entry: (topic_id, label, probability, top_words)
    """
    tokens = preprocess(text)
    bow = dictionary.doc2bow(tokens)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
    topic_dist_sorted = sorted(topic_dist, key=lambda x: x[1], reverse=True)

    results = []
    for topic_id, prob in topic_dist_sorted[:TOP_TOPICS]:
        label = topic_labels.get(str(topic_id), f"Topic_{topic_id}")
        words = [w for w, _ in lda_model.show_topic(topic_id, topn=TOP_WORDS_PER_TOPIC)]
        results.append((topic_id, label, prob, words))
    return results


def display_classification(title, text, results):
    """Pretty-print the classification result for a document."""
    preview = text[:PREVIEW_LENGTH] + ("..." if len(text) > PREVIEW_LENGTH else "")
    print(f"\n{'='*65}")
    print(f"Document : {title}")
    print(f"Preview  : {preview}")
    print(f"{'─'*65}")
    print(f"Top {TOP_TOPICS} topics:")
    for rank, (topic_id, label, prob, words) in enumerate(results, 1):
        top_words_str = ", ".join(words)
        print(f"  #{rank}  Topic {topic_id:2d} | {label:<25} | {prob:.4f} ({prob*100:.1f}%)")
        print(f"       Top words: {top_words_str}")
    print(f"{'='*65}")


print(f"{'='*65}")
print("LOADED TOPICS SUMMARY")
print(f"{'='*65}")
for i in range(num_topics):
    label = topic_labels.get(str(i), f"Topic_{i}")
    words = [w for w, _ in lda_model.show_topic(i, topn=5)]
    print(f"  Topic {i:2d} → {label:<25} [{', '.join(words)}]")

print(f"\n\n{'='*65}")
print("EXAMPLE CLASSIFICATIONS")
print(f"{'='*65}")

for title, text in SAMPLE_DOCUMENTS:
    results = classify(text)
    display_classification(title, text, results)

print("\nInference complete.")
