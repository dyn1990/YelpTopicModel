# -*- coding: utf-8 -*-
"""
Created on Sat May  5 10:10:16 2018

@author: Dyn
"""

import TextClean as tc
from afinn import Afinn
import spacy
afn = Afinn(emoticons=True)
nlp = spacy.load('en', disable=['parser', 'entity', 'ner', 'textcat'])
import re

import time

import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Pool

num_cores = multiprocessing.cpu_count()
num_partitions = num_cores


def parallelize_dataframe(df, func):
    with Pool(processes=num_cores) as pool:
        df = pd.concat(pool.map(func, np.array_split(df, num_partitions)))
    pool.close()
    pool.join()
    return df
 
def partition_clean(data):
    return data.apply(lambda x: tc.clean_text(x))

def partition_afinn(data):
    return data.apply(lambda x: afn.score(x)) 

def partition_feature(data):
    def extract(text):
        return [token for token in nlp(text) if token.tag_ in ['NN', 'NNS']]
    return data.apply(lambda x: str(extract(x)))




if __name__ == '__main__':
    
    business=pd.read_csv("./dataset/yelp_business.csv")
    business_idx = business['categories'].apply(lambda x: True if ('Restaurant' or 'restaurant') in x else False)
    business = business.loc[business_idx]
    
    reviews = pd.read_csv('./dataset/yelp_review.csv')
    reviews = reviews[['review_id', 'business_id', 'text', 'stars']].merge(business[['business_id', 'name']], on='business_id').dropna()
    del business
    
    start = time.time()
    reviews['cleantext'] = parallelize_dataframe(reviews.text, partition_clean)
    print("text cleaning takes {0:.2f} second".format(time.time()-start))
    
    start = time.time()
    reviews['afinn_score'] = parallelize_dataframe(reviews['cleantext'], partition_afinn)
    reviews[['cleantext', 'afinn_score', 'stars']].to_csv("restaurants.csv", index=False)
    print("Afinn sentiment score takes {0:.2f} second".format(time.time()-start))
    
    start = time.time()
    reviews['features'] = parallelize_dataframe(reviews['cleantext'], partition_feature)
    print("Feature extraction takes {0:.2f} second".format(time.time()-start))
    reviews[['cleantext', 'afinn_score', 'features', 'stars']].to_csv("restaurants.csv", index=False)
    
    
    
    
    