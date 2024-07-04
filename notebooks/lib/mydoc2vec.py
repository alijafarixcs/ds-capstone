import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import sys
sys.path.append('.')
import missingno as msno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')




class Doc2VecRecommender:
    def __init__(self,data=None, vector_size=100, window=5, min_count=5, workers=4, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.data=data
        #self.pls=Parallelism(8)
        self.lemmatizer = WordNetLemmatizer()
    def train_similarity(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['all_clear'])
    def get_similars(self,query_desc):
        query_desc =self.preprocess(query_desc)
        query_vec = self.tfidf.transform(query_desc)

        similarity_scores = cosine_similarity(query_vec,self.tfidf_matrix)

        most_similar_idx = similarity_scores.argsort()[:,-100:]

        return most_similar_idx    
    def recommend_by_text(self,search):
        lists=[]
        for i in self.get_similar_indexs(search):
            lists.append(self.data.iloc[i[0]])
        return pd.DataFrame(lists)
    def preprocess(self, text,stop_words_temp=stop_words):
        
        lemmatizer = self.lemmatizer
        sentence = text
        tokens = nltk.word_tokenize(sentence)
        tokens=[word for word in tokens if word not in stop_words_temp]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def train(self, df):
        self.data=df
        #self.pls.set_worker()
        #documents =self.pls.do_paralell(self.data,'all',self.preprocess)
        documents = [TaggedDocument(self.preprocess(doc), [i]) for i, doc in enumerate(df['all'])]
        train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        self.model.build_vocab(train_docs)
        self.model.train(train_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return test_docs
    def load_model(self,path='doc2vec_model'):
         self.model = Doc2Vec.load(path)
    def get_similar_indexs(self,text=["This is a new document to find similar documents for"]):
        text =self.preprocess(text[0])
        new_document_vector = self.model.infer_vector(text)
        similar_docs = self.model.dv.most_similar(new_document_vector, topn=100)

        for doc_index, similarity in similar_docs:
            yield (doc_index, similarity)
    def get_similar_pair(self,df,n):
            for doc_id in range(n):
                inferred_vector = self.model.infer_vector(df[doc_id])
                sims = self.model.dv.most_similar([inferred_vector], topn=10)
                yield (doc_id,sims)
