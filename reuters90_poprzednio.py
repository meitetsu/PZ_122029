from os import listdir
from sklearn import svm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, reuters
import re
import math
cachedStopWords = stopwords.words("english")
min_lenght = 3

class corpus:
    def __init__(self):
        self.documents = []
        self.categories = reuters.categories()
        self.cat_dict = {}
        iterator = 0
        for category in self.categories:
            iterator = iterator + 1
            self.cat_dict[iterator] = category
            for docid in reuters.fileids(category):
                doc_class = iterator
                if docid.startswith("train"):
                    train = 1
                elif docid.startswith("test"):
                    train = 0
                else:
                    raise
                text = reuters.raw(docid)
                doc = document(text, doc_class, train)
                self.add_document(doc)
        self.initialize_vocabulary()
    def add_document(self, document):
        self.documents.append(document)

    def get_train_documents(self):
        train = []
        for doc in self.documents:
            if doc.train == 1:
                train.append(doc.text)
        return train

    def initialize_vocabulary(self):
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        for i,doc in enumerate(self.documents):
            for word in doc.get_unique_words():
                if word not in self.vocabulary:
                    self.vocabulary[i] = word
                    self.inverse_vocabulary[word] = i

    def get_svm_vectors(self,Train = 0, Test = 0):
        Xs = []
        ys = []
        for doc in self.documents:
            if Train == 1 and doc.train == 0:
                continue
            if Test == 1 and doc.train == 1:
                continue
            x = doc.get_vector(self.inverse_vocabulary)
            y = doc.doc_class
            Xs.append(x)
            ys.append(y)
        return (Xs,ys)

class document:
    def __init__(self, text, doc_class = 1, train = 1):
        self.doc_class = doc_class
        self.train = train
        self.text = text
    def preprocessing(self,raw_tokens):
        no_stopwords = [token for token in raw_tokens if token not in cachedStopWords]
        stemmed_tokens = []
        stemmer = PorterStemmer()
        for token in no_stopwords:
            stemmed_tokens.append(stemmer.stem(token))
        p = re.compile('[a-zA-Z]+')
        pattern_checked = []
        for stem in stemmed_tokens:
            if p.match(stem) and len(stem) >= min_lenght:
                pattern_checked.append(stem)
        return pattern_checked
    def get_unique_words(self):
        word_list = []

        for word in self.preprocessing(self.text.split()):
            if not word in word_list:
                word_list.append(word)
        return word_list
    def get_vector(self,inverse_vocabulary):
        lng = len(inverse_vocabulary)
        vector = [0 for i in range(lng)]
        for word in self.preprocessing(self.text.split()):
            vector[inverse_vocabulary[word]] = 1
        return vector


class tf_idf:

    def __init__(self):
        self.D = 0.0
        self.df = {}
    def add_document(self, document):
        self.D += 1.0
        for token in set(document):
            self.df[token] += 1.0
    def idf(self,token):
        return math.log(self.D/self.df[token])
    def tf(self,token,document):
        liczba_wystapien_tokenu = 0.0
        liczba_tokenow = 0.0
        for t in document:
            liczba_tokenow += 1.0
            if t == token:
                liczba_wystapien_tokenu += 1.0
        return liczba_wystapien_tokenu/liczba_tokenow
    def tfidf(self,token, document):
        return self.tf(token,document) * self.idf(token)


klasyfikator = svm.SVC(kernel="linear")
crp = corpus()
(X,y) = crp.get_svm_vectors(Train = 1)
print("starting fitting procedure")
klasyfikator.fit(X,y)
(XT,yt) = crp.get_svm_vectors(Test = 1)
pozytywne = 0
wszystkie = 0
for i,x in enumerate(XT):
    wszystkie += 1
    klasa = klasyfikator.predict(x)
    if klasa == yt[i]:
        pozytywne = pozytywne + 1

print(pozytywne)
print(wszystkie)