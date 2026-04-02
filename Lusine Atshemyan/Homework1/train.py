from sklearn.datasets import fetch_20newsgroups
from gensim.parsing.preprocessing import STOPWORDS
import gensim, os

data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data[:1000]
# print(type(data), '\n'*3, data[:2])
tokenized_data = [gensim.utils.simple_preprocess(text) for text in data]
# print(tokenized_data[:2])
cleared_data = [[word for word in text if word not in STOPWORDS and len(word) >= 3] for text in tokenized_data]
# print(cleared_data[:2])

train_data_dict = gensim.corpora.Dictionary(cleared_data)
train_data_dict.filter_extremes(no_below=5, no_above=0.5)
dataset = [train_data_dict.doc2bow(doc) for doc in cleared_data]

# print(dataset[:2])

lda_model = gensim.models.LdaModel(
    corpus=dataset,
    id2word=train_data_dict,
    num_topics=10,
    passes=15,
    alpha='auto',
    eta='auto',
    random_state=42
)

# topics = lda_model.print_topics(num_words=10)
# for topic in topics:
#     print(topic)

os.makedirs("models", exist_ok=True)
lda_model.save("models/lda_model.model")
train_data_dict.save("models/dictionary.dict")

