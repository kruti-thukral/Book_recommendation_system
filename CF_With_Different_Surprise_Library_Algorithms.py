#!/usr/bin/env python
# coding: utf-8

# Predicting unknown ratings using KNNMeans, KNNZScore, KNNBasic and KNNBaseline algorithms and considering the one with less error

# **Refernces:**
# 
# [1] Yibo Wang, Mingming Wang, and Wei XuA - Sentiment-Enhanced Hybrid Recommender System for Movie Recommendation: A Big Data Analytics Framework: https://www.hindawi.com/journals/wcmc/2018/8263704/
# 
# [2] Susan Li - Building and Testing Recommender Systems With Surprise, Step-By-Step: https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
# 
# [3] Surprise Library: http://surpriselib.com/
# 


# In[ ]:


import sys
import pandas as pd
import numpy as np


# In[ ]:


#download the data by passing the absolute path of file. The file is in gz format, so requires extraction.
path='Complete_Data_new.gz'
data = pd.read_csv(path, compression='gzip')
#data=pd.read_csv('Complete_Data.csv')


# In[ ]:


#data['review_text'].describe()


# In[ ]:


#remove the records that does not have reviews.
data=data[~data['review_text'].isnull()]


# In[7]:


#check the data info
data.info()


# In[ ]:


# Filtering rarely rated books and users from data
min_book_ratings = 15
filter_books = data['book_id'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 15
filter_users = data['user_id'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

df_new = data[(data['book_id'].isin(filter_books)) & (data['user_id'].isin(filter_users))]
#print('The original data frame shape:\t{}'.format(data.shape))
#print('The new data frame shape:\t{}'.format(df_new.shape))


# In[ ]:


#only consider the columns (user_id, book_id, rating) that are required for collaborative filtering
df= df_new[['user_id','book_id','rating']]


# In[ ]:


#len(df.user_id.unique())


# In[ ]:


#len(df.book_id.unique())


# In[ ]:


# Importing required librairs from surprise. 
# We need to define reader object to parse dataframe.
from surprise import Reader,Dataset
reader = Reader()
sdata = Dataset.load_from_df(df[['user_id','book_id','rating']], reader)


# In[ ]:


# Using whole dataset as train set. We make use of build_full_trainset method which builds trainset object.
from surprise.model_selection import train_test_split
trainset = sdata.build_full_trainset()


# In[ ]:


''' We build test set using build_anti_testset() method. 
 Return a list of ratings that can be used as a testset in the test() method.
 The ratings are all the ratings that are not in the trainset, i.e. all the ratings rui where the user u is known, 
 the item i is known, but the rating rui is not in the trainset. 
 As actual rating is unknown, it is either replaced by the fill value or assumed to be equal to the mean of all ratings global_mean'''
testset = trainset.build_anti_testset()


# In[13]:


''' We build the model by making use of KNNWithMeans which is collaborative filtering based algorithm. 
    We are setting minimum number of neighbous (min_k) 10 and maximum number of neighbours (k) = 20  
    We train the model on train set '''
from surprise import KNNWithMeans,KNNBasic,KNNWithZScore,KNNBaseline, accuracy
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo1 = KNNWithMeans(sim_options=sim_options,k=40,min_k=1)
algo1.fit(trainset)
predictions1 = algo1.test(testset)
print("RMSE for KNNMeans:", accuracy.rmse(predictions1, verbose=True))


# In[ ]:


''' We build the model by making use of KNNBasic which is collaborative filtering based algorithm. 
    We are setting minimum number of neighbous (min_k) 1 and maximum number of neighbours (k) = 40  
    We train the model on train set '''

algo2 = KNNBasic(sim_options=sim_options,k=40,min_k=1)
algo2.fit(trainset)

predictions2 = algo2.test(testset)
print("RMSE for KNNBasic:", accuracy.rmse(predictions2, verbose=True))


# In[ ]:


''' We build the model by making use of KNNBasic which is collaborative filtering based algorithm. 
    We are setting minimum number of neighbous (min_k) 1 and maximum number of neighbours (k) = 40  
    We train the model on train set '''

algo3 = KNNBaseline(sim_options=sim_options,k=40,min_k=1)
algo3.fit(trainset)

predictions3 = algo3.test(testset)
print("RMSE for KNNBaseline:", accuracy.rmse(predictions3, verbose=True))


# In[ ]:


''' We build the model by making use of KNNBasic which is collaborative filtering based algorithm. 
    We are setting minimum number of neighbous (min_k) 1 and maximum number of neighbours (k) = 40  
    We train the model on train set '''

algo4 = KNNWithZScore(sim_options=sim_options,k=40,min_k=1)
algo4.fit(trainset)

predictions4 = algo4.test(testset)
print("RMSE for KNNBasic:", accuracy.rmse(predictions4, verbose=True))

