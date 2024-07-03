import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import sys
sys.path.append('.')
from lib.paralellism import *

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import torch
from torch.utils.data import DataLoader, Dataset
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split

class Doc2VecDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

class Doc2VecTrainer:
    def __init__(self, vector_size, window, min_count, workers, epochs):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def preprocess(self, doc):
        return [word for word in doc.split() if word not in stop_words]

    def train(self, df):
        self.data = df
        documents = [TaggedDocument(self.preprocess(doc), [i]) for i, doc in enumerate(df['all'])]
        train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)

        # Create DataLoader for training
        train_dataset = Doc2VecDataset(train_docs)
        # Assuming len(train_docs) is 128
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)  # Divisible by 128



        # Initialize Doc2Vec model
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        self.model.build_vocab(train_docs)

        # Move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.wv.vectors = torch.tensor(self.model.wv.vectors).to(device)

        for epoch in range(self.epochs):
            for i in range(0, len(train_dataset), 10000):
                batch = train_dataset[i:i+10000]
        # Training loop
        """for epoch in range(self.epochs):
            for batch in train_loader:
                # Move batch to GPU
                batch = [doc.to(device) for doc in batch]"""
        self.model.train(batch, total_examples=len(batch), epochs=self.epochs)

        return test_docs
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

    def recommend_by_text(self,search=[]):
        lists=[]
        for i in self.get_similar_indexs([search]):
            lists.append(self.data.iloc[i])
        return pd.DataFrame(lists)
    def preprocess(self, text,stop_words_temp=stop_words.copy()):
        return [word for word in text.split() if word not in stop_words_temp]

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
        
        new_document_vector =self.model.infer_vector(text)
        similar_docs =self.model.dv.most_similar(new_document_vector, topn=10)
        for doc, similarity in similar_docs:
             yield (doc,similarity)
    def get_similar_pair(self,df,n):
            for doc_id in range(n):
                inferred_vector = self.model.infer_vector(df[doc_id])
                sims = self.model.dv.most_similar([inferred_vector], topn=10)
                yield (doc_id,sims)
