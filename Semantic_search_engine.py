import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')


DOCUMENTS = [
    "The Python programming language is excellent for data science and machine learning.",
    "Natural Language Processing (NLP) involves training models to understand human text.",
    "Data science tools like Pandas and NumPy are essential for any analysis task.",
    "A machine learning engineer needs strong skills in Python, NumPy, and Scikit-learn.",
    "NLP engineers often work with text classification and semantic search algorithms."
]

QUERY = "I need Python skills for NLP"

ENGLISH_STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """Cleans text by lowering case, removing punctuation, and filtering stopwords."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def run_semantic_search(corpus, query):
    """
    Performs vectorization using TF-IDF and calculates cosine similarity 
    between the query and the documents.
    """
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]
    preprocessed_query = preprocess_text(query)
    
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)
    
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    print(f"--- Search Results for Query: '{query}' ---\n")
    print(f"| Rank | Similarity Score | Document Content |")
    print(f"|------|------------------|------------------|")
    
    for rank, doc_index in enumerate(ranked_indices):
        score = similarity_scores[doc_index]
        document = corpus[doc_index]
        
        if score > 0:
            score_str = f"{score:.4f}"
            print(f"| {rank + 1:<4} | {score_str:<16} | {document[:60]}... |")
        else:
            print(f"No further relevant documents found (Score: 0.0000)")
            break

if __name__ == "__main__":
    run_semantic_search(DOCUMENTS, QUERY)
