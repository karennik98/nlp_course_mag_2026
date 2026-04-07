LDA Topic Modeling Project
Այս նախագիծը իրականացնում է թեմաների մոդելավորում (Topic Modeling) օգտագործելով LDA (Latent Dirichlet Allocation) ալգորիթմը և Gensim գրադարանը:

Ծրագրի կառուցվածքը
train_lda.py - Մարզում է LDA մոդելը 20 Newsgroups տվյալների բազայի վրա:
label_topics.py - Թույլ է տալիս ինտերակտիվ կերպով անուններ տալ գտնված թեմաներին:
inference.py - Օգտագործում է պատրաստի մոդելը նոր տեքստերը դասակարգելու համար:
models/ - Այս թղթապանակում պահվում են մարզված մոդելը և բառարանը:
Ինչպես աշխատեցնել
Տեղադրել անհրաժեշտ գրադարանները:
pip install gensim scikit-learn
