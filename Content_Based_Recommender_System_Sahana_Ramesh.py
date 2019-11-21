
#importing libraries
import sys
import pandas as pd
import os

#reading the dataset into a Dataframe
df=pd.read_csv('Complete_Data.csv')

#View the first 5 rows
df.head()

#Get the dataset info on number of rows and columns and each column information
df.info()

#take only the required columns and store it in another dataset
content_data = df[['user_id', 'book_id', 'country_code', 'language_code', 'rating', 'authorid', 'description', 'title']]

content_data.head()

content_data.info()

#Taking only the book metadata as we are doing cntent based and we do not need user information
book_data = content_data[['book_id', 'description', 'title', 'country_code', 'language_code', 'authorid']]

book_data

#Since we have duplicate bookids, we do a groupby on bookid
book_data = book_data.groupby(['book_id']).agg(lambda x: list(x)).reset_index()

#We are getting same values for title, description, country code, language code after groupby as it is the same bookid. So we take only one value for each
book_data['title'] = book_data['title'].apply(lambda x: x[:1])
book_data['description'] = book_data['description'].apply(lambda x: x[:1])
book_data['country_code'] = book_data['country_code'].apply(lambda x: x[:1])
book_data['language_code'] = book_data['language_code'].apply(lambda x: x[:1])

#Viewing the first 5 records of book_data
book_data.head()

book_data

#We need to combine the columns description, authorid, country code and language code to apply tf-idf on these columns as a whole. We have to convert it to string as we cannot combine on list
book_data = book_data.applymap(str).reset_index()

#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['description']) + ' ' + ' '.join(x['authorid']) + ' ' + x['country_code'] + ' ' + ' '.join(x['language_code'])

#We are ceating a soup out the desired metadata and storing it in another column named soup
book_data['soup'] = book_data.apply(create_soup, axis=1)

#book_data = book_data.head(50000)

#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object
tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words='english')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tf.fit_transform(book_data['soup'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Construct a reverse mapping of indices and book titles, and drop duplicate titles, if any
indices = pd.Series(book_data.index, index=book_data['title']).drop_duplicates()

# Function that takes in book title as input and gives recommendations 
def content_recommender(title, cosine_sim=cosine_sim, df=book_data, indices=indices):
    # Obtain the index of the book that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books. Ignore the first book.
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    # return df['title'].iloc[book_indices]
    book =[]
    score = []
    index=[]
    book = df['title'].iloc[book_indices]
    index,score = map(list,zip(*sim_scores))
    recommend_metadata=pd.DataFrame({'book' : book, 'Score' : score})
    print(recommend_metadata)
    result1 = recommend_metadata
    recommend_metadata.to_csv("result1.csv", index=False)
    #return df['title'].iloc[book_indices]

#Get recommendations for Maison Ikkoku, Volume 2 (Maison Ikkoku, #2)
content_recommender("['Maison Ikkoku, Volume 2 (Maison Ikkoku, #2)']")

#Get recommendations for Inu Yasha Animanga, Volume 17
content_recommender("['Inu Yasha Animanga, Volume 17']")

#Get recommendations for Emma, Vol. 01
content_recommender("['Emma, Vol. 01']")

#Get recommendations for Black Orchid
content_recommender("['Black Orchid']")

#Get all the records while has the title "Inu Yasha Animanga, Volume 17"
y = df[df['title'].astype(str).str.contains("Inu Yasha Animanga, Volume 17")]

y

#Pick a user who has read Inu Yasha Animanga, Volume 17 and get the list of all his books just to compare his list contains some of the recommendd books for Inu Yasha Animanga, Volume 17
df[df['user_id']=='babc631ce1f8d2deab56f65cee762062']['title'].values

#Get the list of all the books this user has read
result2 = df[df['user_id']=='babc631ce1f8d2deab56f65cee762062']['title'].values

result0 = pd.DataFrame(result2)

result0['book'] = result0

result0['book']

#get the list of recommendations for book Inu Yasha Animanga, Volume 17 from the result1.csv
result1 = df=pd.read_csv('result1.csv')

result1.info()

result1['book'] = result1['book'].str[1:-1]

result1['book']
#Check if the recommendation list contains the books from the user list of books
d1=result0['book'].isin(result1['book'])

#d1

#d2 = result1[d1]

#d2
