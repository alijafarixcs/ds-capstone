
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class mltoos_recommender:
 

  def __init__(self, arrays):

    self.arrays = arrays
    self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    self.tfidf_matrix = self.tfidf.fit_transform(arrays)
    

  def get_similars(self,query_desc,tfidf,tfidf_matrix):
        query_vec = tfidf.transform([query_desc.lower()])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix)
        most_similar_idx = similarity_scores.argsort()[0][-100:]
        return most_similar_idx
  def get_similarities(self,query):
    indexss=self.get_similars(query,self.tfidf,self.tfidf_matrix)
    return indexss
  
  def show_display(show=True):
    if show:
        pd.set_option('display.max_colwidth', None)
    else:
        pd.set_option('display.max_colwidth', -1)
    


    



