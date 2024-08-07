# Creating a simple search engine

## Goal of this notebook:

- Understand different techniques in general: Explore a few different ways we can implement a simple search engine for queries. The goal is that the user can type a query related to the zoomcamp course FAQ pages, and can receive a few results in order of their relevance. We will see how different methods yield different results and which are more effective in extracting the most relevant results. 

- Explore different vector search methods: Using methods like CountVectorizer/TfidfVectorizer, and then slightly more sophisticated methods using singular value reduction (dimensionality reduction methods) like SVD and NMF to embed the vector and gain semantic meaning.
-----------------------------------------
## Text vs Vector Search

In this exercise we will look at both __Text Search and Semantic/Vector__ search methods. Note that both these methods are under the umbrella of the 'Bag of Words' method, which means that the order of the words has no meaning. This has obvious limitations and can be overcome with more advanced neural networks like BERT. 

We can illustrate the difference in these methods with a small example:

`query = 'I just discovered the course. Can I still join?'`

In text search, we will find all the documents that contain words like 'discovered', 'course', 'join', etc. However, often the user forms a question that does not really match the documents. For example:

`query = 'I just found out about the program. Can I still enroll?'`

Semantically, both queries have the same meaning, but with text search we will not get good results. This is when a semantic/vector approach will perform much better. 

## Vector Search Methods

Here is a quick breakdown of the steps of each of these methods:

__Text search methods__:
- create an instance of the Vectorizer (CV, Tfdif), fit_transform the documents to get document matrix (X), and transform the query (q)
- calculate similarity score (with cosine similarity between X and q) and rank results

__Semantic/Vector methods__:
- create an instance of the Vectorizer (CV, Tfdif), fit_transform the documents to get document matrix (X), and transform the query (Q)
- create an instance of the Embedder (SVD, NMF), fit_transform X to dense term-document matrix (X_emb), and transform Q to get dense query array (Q_emb)
- calculate similarity score (with cosine similarity between X_emb and Q_emb) and rank results


