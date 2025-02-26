import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# Loading the research paper dataset
try:
    df = pd.read_csv("data/cleaned_arxiv_data.csv")
    print("Dataset loaded successfully.")
    print("Columns in dataset:", df.columns)
except Exception as e:
    print("Error loading dataset:", e)
    raise

# Ensuring required columns exist
required_columns = ["titles", "summaries", "terms"]  # Updated to match dataset
if not all(column in df.columns for column in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# Preprocessing
stop_words = set(stopwords.words("english"))
df["summaries"] = df["summaries"].fillna("").astype(str)  # Ensure summaries are strings
df["processed_text"] = df["summaries"].apply(lambda x: " ".join([word for word in x.lower().split() if word not in stop_words]))

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

def get_recommendations(query, top_n=5):
    try:
        query_tfidf = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
        top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        
        recommendations = df.iloc[top_indices][["titles", "summaries", "terms"]].to_dict(orient="records")
        return recommendations
    except Exception as e:
        print("Error in get_recommendations:", e)
        return []