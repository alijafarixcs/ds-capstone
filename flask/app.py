
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import random 
import sys
sys.path.append('.')
sys.path.append('../notebooks')
sys.path.append('../notebooks/lib')

from notebooks import *
from notebooks.lib import *
from notebooks.lib.data_prepration import DataPreparation
from notebooks.lib.mydoc2vec import  Doc2VecRecommender
from notebooks.lib.paralellism import  Parallelism
from notebooks.lib.cleaning import *


import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk

from gensim.models import Doc2Vec
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
#!pip install scipy==1.12
SEED = 448

file_path = 'data\cleared_columns_title.csv'
data_prep = DataPreparation(file_path)
data_prep.read_large_csv()
random.seed(SEED)
_doc2vec_load = Doc2VecRecommender(data_prep.data)
_doc2vec_load.load_model(r'notebooks\models\doc2vec_model_title')
_doc2vec_load.train_similarity()

from flask import Flask, render_template,request, url_for

app = Flask(__name__, static_url_path='',template_folder="template_folder", static_folder='static')



@app.route('/predictsimilar', methods=['POST'])
def predictsimilar():
    text = request.form['text']

    indexs=_doc2vec_load.get_similars(text)
    indexs=data_prep.data.loc[indexs[0]][['index','title']]
    return render_template('results.html',column_names=indexs.columns.values, row_data=list(indexs.values.tolist()), zip=zip)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    indexs=_doc2vec_load.recommend_by_text([text])[['index','title']]

    return render_template('results.html',column_names=indexs.columns.values, row_data=list(indexs.values.tolist()), zip=zip)

@app.route('/index')
def index():
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML System Test</title>
        </head>
        <body>
            <h1>Test Your ML System</h1>
            <form method="POST" action="/predict">
                <label for="text">Enter text to test:</label><br>
                <input type="text" id="text" name="text" rows="5">
                <button type="submit">Predict</button>
            </form>
                  <form method="POST" action="/predictsimilar">
                <label for="text">Enter text to test:</label><br>
                <input type="text" id="text" name="text" rows="5">
                <button type="submit">predictsimilar</button>
            </form>
        </body>
        </html>
    """

if __name__ == '__main__':
    #app.debug=True
    app.run()

