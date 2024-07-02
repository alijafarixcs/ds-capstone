import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import pandarallel
from pandarallel import pandarallel

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import random 

import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
#!pip install scipy==1.12
SEED = 448
#!pip install  pandarallel


class Parallelism:
    def __init__(self, n_worker):
        self.n_worker=n_worker
        self._library = None
        self._library_name = "<pandarallel>"  # Replace with the library name

    def _check_and_install_library(self):
        try:
            self._library = importlib.import_module(self._library_name)
        except ModuleNotFoundError:
            # Install the library using pip (assuming it's available)
            import subprocess
            subprocess.run(["pip", "install", self._library_name])
            self._library = importlib.import_module(self._library_name)
            

    def set_worker(self):
        pandarallel.initialize(nb_workers=self.n_worker)


        

    
    def do_paralell(self,df,column,func):
        return df[column].parallel_apply(func)