One data science application that is used by nearly every application or website today is a recommendation system.
A recommendation system is a common tool used by websites nowadays to suggest articles to its users.

In this article, I will walk you through how to build an Article Recommendation System with Machine Learning using Python.

There are different ways to build a recommendation system like: 
Collaborative Filtering
Content-Based Filtering
Hybrid Filtering
User-Based Filtering
Item-Based Filtering

For article recommendation we need to focus on content rather than user interest.
To recommend articles based on the content, we need to understand the content of the article, match the content with all the other articles and recommend the most suitable articles for the article that the reader is already reading
We will use the concept of Cosine Similarity. Cosine similarity is a method of building recommendation systems based on the content. It is used to find similarities between two different pieces of text documents.

To create an article recommendation system, I collected data about some of the articles , you can find them in this repository named articles(1).cvs

So let’s import the necessary Python libraries and the dataset we need to create an articles recommendation system
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("article.cvs", encoding='latin1')
data.head()

Let us now use the cosine similarity algorithm and write a Python function to recommend articles.

articles = data["Article"].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
def recommend_articles(x):
    return ", ".join(data["Title"].loc[x.argsort()[-5:-1]])    
data["Recommended Articles"] = [recommend_articles(x) for x in uni_sim]
data.head()

As you can see from the output of the above code, a new column has been added to the dataset that contains the titles of all the recommended articles. 
Now let’s see all the recommendations for an article by:

print(data["Recommended Articles"][16])

This prints the recommended article for index 16

This is how you can build an article recommendation system.
