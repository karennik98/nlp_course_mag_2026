import os
from sklearn.datasets import fetch_20newsgroups
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# 1. Ստեղծել models թղթապանակը, եթե չկա
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Բեռնել տվյալները
print("Loading data...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')).data

# 3. Նախամշակում
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) >= 3:
            result.append(token)
    return result

print("Preprocessing documents...")
processed_docs = [preprocess(doc) for doc in data]

# 4. Բառարան և Կորպուս
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 5. LDA Մարզում (10 թեմա)
print("Training LDA model...")
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=15, alpha='auto', eta='auto')

# 6. Պահպանում
lda_model.save('models/lda_model.model')
dictionary.save('models/dictionary.dict')

print("Model trained and saved successfully!")

# Ցուցադրել թեմաները
print("\nDiscovered Topics:")
for idx, topic in lda_model.print_topics(-1, num_words=10):
    print(f"Topic {idx}: {topic}")
