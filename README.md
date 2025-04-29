# Recommendation-systems

In the context of IDF (Inverse Document Frequency), "downscales" means that it reduces the importance or weight of words that appear frequently across documents. This is done to give more weight to words that are less common and potentially more informative for distinguishing between documents.

One of the original and straightforward methods for transforming texts into numerical values is TF-IDF. It's used to convert a collection of raw documents to a matrix of TF-IDF features. Let's break this down:

TF (Term Frequency): this summarizes how often a given word appears within a document
IDF (Inverse Document Frequency): this downscales words that appear a lot across documents. A word that appears in many documents will not be a good keyword to categorize these documents because it does not help differentiate them
TF-IDF are word frequency scores that try to highlight words that are more interesting, such as frequent in a document but not across documents. The TF-IDF value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

In sklearn the TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features. It's equivalent to CountVectorizer followed by TfidfTransformer.

*Compute similarities*

We need a similarity measure to compute how similar each Movie is to every other Movie. A common choice is the cosine similarity, which measures the cosine of the angle between two vectors. The closer the cosine similarity is to 1, the more similar the items are.

TL;DR:

The function should take the title of a Movie and output recommendations. To get recommendations, we will get the index of the Movie in our DataFrame, get the list of cosine similarity scores for that Movie, and then sort the scores to get the indexes of the Movies with the highest similarity. We can then use these indexes to get the recommended Movies from our DataFrame.
