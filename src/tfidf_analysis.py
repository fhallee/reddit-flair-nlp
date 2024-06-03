from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3

def main():
    # Initialize tokenizer, lemmatizer, stopwords, and TF-IDF vectorizer
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))

    # Connect to the SQLite database and get flairs
    conn = sqlite3.connect("../data/posts.db")
    c = conn.cursor()
    sql = f"SELECT DISTINCT flair FROM posts"
    data = c.execute(sql).fetchall()
    flairs = [row[0] for row in data]

    # Preprocess text for each flair
    documents = []
    for flair in flairs:
        sql = f"SELECT text FROM posts WHERE flair = '{flair}'"
        data = c.execute(sql).fetchall()
        text_data = ''.join([row[0] for row in data]) 
        tokens = tokenizer.tokenize(text_data) # Tokenize text
        filtered_tokens = [token for token in tokens if token not in stop_words] # Remove stopwords
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens] # Lemmatize text
        document_text = ' '.join(lemmatized_tokens)
        documents.append(document_text)

    # Fit TF-IDF vectorizer to the preprocessed documents
    vectorizer.fit_transform(documents)
    # Get feature names (bigrams)
    feature_names = vectorizer.get_feature_names_out()
    # Get TF-IDF weights for each document
    tfidf_weights = vectorizer.transform(documents).toarray()

    # Print the top 15 informative features (bigrams) for each document
    for i, doc in enumerate(documents):
        print(f"\nDocument: {flairs[i]}")
        top_features = sorted(zip(feature_names, tfidf_weights[i]), key=lambda x: x[1], reverse=True)[:15]
        print("Top 15 Informative Features (TF-IDF):")
        for feature, score in top_features:
            print(f"- {feature}: {score:.4f}")

    conn.close()

if __name__ == "__main__":
    main()