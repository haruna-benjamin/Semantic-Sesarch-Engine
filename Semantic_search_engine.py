import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download stop words if not already present
# This is required for the text cleaning step
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- CONFIGURATION & CORPUS ---

# 1. Document Corpus: A small set of documents to search through
DOCUMENTS = [
    "The Python programming language is excellent for data science and machine learning.",
    "Natural Language Processing (NLP) involves training models to understand human text.",
    "Data science tools like Pandas and NumPy are essential for any analysis task.",
    "A machine learning engineer needs strong skills in Python, NumPy, and Scikit-learn.",
    "NLP engineers often work with text classification and semantic search algorithms."
]

# 2. Query to search against the corpus
QUERY = "I need Python skills for NLP"

# Initialize stop words
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

# --- HELPER FUNCTIONS ---

def preprocess_text(text):
    """Cleans text by lowering case, removing punctuation, and filtering stopwords."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# --- MAIN ENGINE ---

def run_semantic_search(corpus, query):
    """
    Performs vectorization using TF-IDF and calculates cosine similarity 
    between the query and the documents.
    """
    # 1. Preprocess all documents and the query
    preprocessed_corpus = [preprocess_text(doc) for doc in corpus]
    preprocessed_query = preprocess_text(query)
    
    # 2. Initialize the TF-IDF Vectorizer
    # This transforms text into numerical vectors
    vectorizer = TfidfVectorizer()
    
    # 3. Fit the vectorizer to the corpus and transform the documents
    # 'fit' learns the vocabulary (all unique words)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_corpus)
    
    # 4. Transform the query using the SAME vectorizer
    # Crucially, we use 'transform' only, to apply the corpus's vocabulary to the query
    query_vector = vectorizer.transform([preprocessed_query])
    
    # 5. Calculate Cosine Similarity
    # This scores the query vector against every document vector
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 6. Rank the results
    # Get the indices that would sort the scores in descending order
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    print(f"--- Search Results for Query: '{query}' ---\n")
    print(f"| Rank | Similarity Score | Document Content |")
    print(f"|------|------------------|------------------|")
    
    # 7. Print the ranked results
    for rank, doc_index in enumerate(ranked_indices):
        score = similarity_scores[doc_index]
        document = corpus[doc_index]
        
        # Only show results with positive similarity
        if score > 0:
            score_str = f"{score:.4f}"
            print(f"| {rank + 1:<4} | {score_str:<16} | {document[:60]}... |")
        else:
            print(f"No further relevant documents found (Score: 0.0000)")
            break

# Run the search engine
if __name__ == "__main__":
    run_semantic_search(DOCUMENTS, QUERY)
