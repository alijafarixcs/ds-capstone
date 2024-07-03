import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def read_large_csv(self,file_path, chunksize=100000):
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunks.append(chunk)
        
        self.data = pd.concat(chunks, ignore_index=True)
        return self.data
    
    def clean_and_preprocess(self,large_size=False):
        
        if not large_size:
            self.data = pd.read_csv(self.file_path)
        else:
            self.data =self.read_large_csv(self.file_path,10000)
        remove_cols=['Unnamed: 0.1', 'Unnamed: 0',]
        if all(col in self.data.columns for col in remove_cols):
            self.data = self.data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
        
        
        self.data['Price'] = self.data['Price'].fillna(np.mean(self.data['Price']))
        self.data['ratingsCount'] = self.data['ratingsCount'].fillna(np.mean(self.data['ratingsCount']))
        self.data=self.data.dropna(subset=['Title', 'description'])
        self.data['authors'] = self.data['authors'].fillna('[]')
        self.data['categories'] = self.data['categories'].fillna('[]')
        self.data['publishedDate'] = pd.to_datetime(self.data['publishedDate'], errors='coerce')
        self.data['publishedDate'] = self.data['publishedDate'].ffill(axis=0)
        self.data['publishedDate'] = self.data['publishedDate'].bfill(axis=0)
        self.data['Title']=self.data['Title'].astype(str)
       
        
        self.data['publishedDate'] = self.data['publishedDate'].replace({pd.NaT:np.nan})
        self.data['review/time'] = pd.to_datetime(self.data['review/time'], unit='s', errors='coerce')
        
        
        self.data[['helpful_votes', 'total_votes']] = self.data['review/helpfulness'].str.split('/', expand=True).astype(int)
        self.data['review/helpfulness']=self.data['helpful_votes']/self.data[ 'total_votes']
        self.data = self.data.drop(columns=['helpful_votes', 'total_votes'])
        self.data['authors'] = self.data['authors'].apply(ast.literal_eval)
        self.data['categories'] = self.data['categories'].apply(ast.literal_eval)
        new_column_names = [col.lower().replace('review/', '') for col in self.data.columns]
        self.data.rename(columns=dict(zip(self.data.columns, new_column_names)),inplace=True)
        self.data[~self.data.duplicated(subset=['description', 'title'], keep='first')]
        self.data.fillna('',inplace=True)
    def Normalize(self):
        self.data=self.data.applymap(lambda x: x.lower() if type(x) is str else x)
    def generate_plots(self):
        sns.set(style="whitegrid")

        # 1. Distribution of Review Scores
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.data,x='score', color='skyblue')
        plt.title('Distribution of Review Scores')
        plt.xlabel('Review Score')
        plt.ylabel('Frequency')
        plt.show()

        # 2. Most Reviewed Authors
        authors_exploded = self.data.explode('authors')
        author_counts = authors_exploded['authors'].value_counts().head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=author_counts.values, y=author_counts.index, palette='viridis')
        plt.title('Top 10 Most Reviewed Authors')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Authors')
        plt.show()

        # 3. Top Categories
        categories_exploded = self.data.explode('categories')
        category_counts = categories_exploded['categories'].value_counts().head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=category_counts.values, y=category_counts.index, palette='magma')
        plt.title('Top 10 Book Categories')
        plt.xlabel('Number of Books')
        plt.ylabel('Categories')
        plt.show()

  


