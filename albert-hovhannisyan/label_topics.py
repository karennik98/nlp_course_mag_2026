import json
import os
from gensim import corpora
from gensim.models import LdaModel

MODELS_DIR = "models"
LABELS_FILE = os.path.join(MODELS_DIR, "topic_labels.json")
TOP_N_WORDS = 20

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
print(f"Loaded model with {num_topics} topics.\n")

print(f"{'='*65}")
print(f"ALL TOPICS (top {TOP_N_WORDS} words with probabilities)")
print(f"{'='*65}")
for topic_id in range(num_topics):
    words = lda_model.show_topic(topic_id, topn=TOP_N_WORDS)
    print(f"\nTopic {topic_id:2d}:")
    for word, prob in words:
        print(f"  {word:<20} {prob:.4f}")

print(f"\n{'='*65}")
print("TOPIC LABELING")
print("Enter a meaningful name for each topic (press Enter to skip).")
print(f"{'='*65}\n")

topic_labels = {}
for topic_id in range(num_topics):
    words = lda_model.show_topic(topic_id, topn=5)
    top_words = ", ".join(w for w, _ in words)
    default_name = f"Topic_{topic_id}"
    label = input(f"Topic {topic_id:2d} [{top_words}]\n  Name (default: '{default_name}'): ").strip()
    topic_labels[str(topic_id)] = label if label else default_name
    print()

os.makedirs(MODELS_DIR, exist_ok=True)
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(topic_labels, f, indent=2, ensure_ascii=False)
print(f"Topic labels saved to '{LABELS_FILE}'")

print(f"\n{'='*65}")
print("FINAL TOPIC SUMMARY")
print(f"{'='*65}")
for topic_id in range(num_topics):
    label = topic_labels[str(topic_id)]
    words = lda_model.show_topic(topic_id, topn=5)
    top_words = ", ".join(w for w, _ in words)
    print(f"  Topic {topic_id:2d} → {label:<25} [{top_words}]")

print("\nDone. Run 'inference.py' to classify new documents.")
