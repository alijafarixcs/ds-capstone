import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import random 


def normalize_column(df, col):
  try:
    df[col] = df[col].str.split('/', expand=True)[0].astype(float)
    df[col] = df[col] / df[col.name + '_denom']
  except (AttributeError, ValueError):
    pass
  df[col] = df[col].fillna(0).astype(float)
  return df


def fill_find_by_book(df_sorted, find=["publisher"], by=["title"]):
  c=0
  df_list=[]
  
  for index, row in df_sorted.iterrows():
    for f, b in zip(find, by):
      if pd.isna(row[f]):
        same_pubs = df_sorted.loc[(df_sorted[b] == row[b]) & (df_sorted.index != index) & (df_sorted[f].notna())]
        if len(same_pubs) > 0:
          matching_row = same_pubs.iloc[0]
          row[f] = matching_row[f]
    df_list.append(row)

  return df_list

