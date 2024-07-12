import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file
data = pd.read_csv(r"C:\Users\HP 840\Desktop\Article Recommendation\articles (1).csv", encoding='latin1')

# Display the first few rows of the dataframe
print(data.head())

# Extract the articles from the dataframe
articles = data["Article"].tolist()

# Create the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(articles)
 
# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Define a function to recommend articles
def recommend_articles(x):
    return ", ".join(data["Title"].iloc[x.argsort()[-5:-1]])

# Create a new column with recommended articles
data["Recommended Articles"] = [recommend_articles(x) for x in cosine_sim_matrix]

# Display the first few rows of the dataframe with the recommendations
print(data.head())

print(data["Recommended Articles"][22])