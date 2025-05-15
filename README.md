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


---

# 🎬 Highly Robust Movie Recommendation Engine

A modular and scalable recommendation system that combines collaborative filtering, content-based filtering, and hybrid techniques to provide personalized movie recommendations. Designed as a learning and demonstration project to showcase real-world recommender system pipelines.

---

## 📌 Features

* ✅ **Collaborative Filtering** using user-item interaction matrix
* ✅ **Content-Based Filtering** using genre, cast, crew, and metadata
* ✅ **Hybrid Recommender** that combines multiple approaches
* ✅ **Knowledge-Based Recommender** with user input logic
* 🛠️ Modular Python scripts for each recommender technique
* 📦 Built with `pandas`, `scikit-learn`, `surprise`, and more

---

## 📁 Project Structure

```bash
Highly-Robust-Movie-Recommendation-engine/
│
├── data/                    # Raw & processed datasets
├── Collaborative Filtering.py
├── Content Based Recommenders.py
├── Hybrid Recommender.py
├── Knowledge Based Recommender.py
├── utils/                   # Helper functions (optional)
└── README.md
```

---

## 🧠 Algorithms Used

| Type                    | Technique                                            |
| ----------------------- | ---------------------------------------------------- |
| Collaborative Filtering | User-Based & Item-Based (with Surprise library)      |
| Content-Based Filtering | TF-IDF + Cosine Similarity on genres, cast, overview |
| Hybrid Approach         | Weighted combination or switching logic              |
| Knowledge-Based         | Simple rules using user preferences                  |

---

## 🧪 Evaluation Metrics

* RMSE for collaborative models
* Precision, Recall



---

## 📊 Dataset

* **Source**: [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* **Fields Used**: title, genres, overview, keywords, cast, crew, ratings


---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Bhasha03/Highly-Robust-Movie-Recommendation-engine.git

# Navigate into project
cd Highly-Robust-Movie-Recommendation-engine

# (Optional) Setup virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run scripts
python Collaborative\ Filtering.py
python Content\ Based\ Recommenders.py
```

---

## 📈 Example Output

```
Input: User ID 237
Top 5 recommended movies:
1. Inception
2. The Matrix
3. Interstellar
4. The Dark Knight
5. Fight Club
```

---

## 🎯 Future Work

* ✅ Add web UI using Streamlit
* 📊 Add evaluation & benchmarking
* 🔍 Hyperparameter tuning for Surprise models
* 🧪 A/B testing logic for hybrid blending

---

## 👨‍💻 Author

Bhavani Shankar
An aspiring Machine Learning Engineer passionate about building real-world intelligent systems.

