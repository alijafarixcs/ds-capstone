import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class Doc2VecRecommender:
    def __init__(self,data=None, vector_size=100, window=5, min_count=2, workers=4, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.data=data
    def recommend_by_text(self,search=[]):
        lists=[]
        for i in self.get_similar_indexs([search]):
            lists.append(self.data.iloc[i[0]])
        return pd.DataFrame(lists)
    def preprocess(self, text):
        return [word for word in text.split() if word.lower() not in stop_words]

    def train(self, df):
        documents = [TaggedDocument(self.preprocess(doc), [i]) for i, doc in enumerate(df['all'])]
        train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        self.model.build_vocab(train_docs)
        self.model.train(train_docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    def load_model(self,path='doc2vec_model'):
         self.model = Doc2Vec.load(path)
    def get_similar_indexs(self,text=["This is a new document to find similar documents for"]):
        
        new_document_vector =self.model.infer_vector(text)
        similar_docs =self.model.dv.most_similar(new_document_vector, topn=10)
        for doc, similarity in similar_docs:
             yield (doc,similarity)
    def get_similar_pair(self,df,n):
            for doc_id in range(n):
                inferred_vector = self.model.infer_vector(df[doc_id])
                sims = self.model.dv.most_similar([inferred_vector], topn=10)
                yield (doc_id,sims)
