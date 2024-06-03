import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
import random
import sqlite3

def main():
    # Initialize tokenizer, lemmatizer, stopwords, and TF-IDF vectorizer
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(max_features=5000)

    # Set random seed for reproducible results
    np.random.seed(1)
    random.seed(1)

    # Connect to the SQLite database and read it into a DataFrame
    conn = sqlite3.connect("../data/posts.db")
    sql = f"SELECT text, flair FROM posts"
    corpus = pd.read_sql(sql, conn)
    conn.close()

    # Extract text and labels
    corpus_text = corpus['text'].tolist()
    labels = corpus['flair'].tolist()

    # Create a list of crisis posts
    crisis_text = [text for text, label in zip(corpus_text, labels) if label == 'Crisis']
    # Create a list of non-crisis texts
    non_crisis_text = [text for text, label in zip(corpus_text, labels) if label == 'Non_crisis']

    # Balance the dataset using random undersampling
    crisis_count = len(crisis_text)
    random.shuffle(non_crisis_text)
    balanced_data = {'text': crisis_text + non_crisis_text[:crisis_count], 'flair': ['Crisis'] * crisis_count + ['Not crisis'] * crisis_count}
    corpus = pd.DataFrame(balanced_data)

    # Preprocess text
    for index, entry in corpus['text'].items():
        text = entry.lower()
        tokens = tokenizer.tokenize(text)
        filtered_words = [token for token in tokens if token not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_words]
        corpus.loc[index, 'text_final'] = str(lemmatized_tokens)

    # Split data into train and test sets
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'],corpus['flair'],test_size=0.3, stratify=corpus['flair'])

    # Encode labels
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Transform text data using TF-IDF vectorizer
    vectorizer.fit(corpus['text_final'])
    Train_X_Tfidf = vectorizer.transform(Train_X)
    Test_X_Tfidf = vectorizer.transform(Test_X)

    # Train Dummy Classifier (baseline model)
    dummy_clf = DummyClassifier()
    dummy_clf.fit(Train_X_Tfidf,Train_Y)
    predictions_dummy = dummy_clf.predict(Test_X_Tfidf)

    # Train Naive Bayes Classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    predictions_NB = Naive.predict(Test_X_Tfidf)

    # Train Support Vector Machine Classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(Train_X_Tfidf,Train_Y)
    predictions_clf = clf.predict(Test_X_Tfidf)

    # Define function to calculate and print metrics
    def print_metrics(model_name, predictions):
        accuracy = accuracy_score(Test_Y, predictions)
        precision = precision_score(Test_Y, predictions, pos_label=0)
        recall = recall_score(Test_Y, predictions, pos_label=0)
        f1 = f1_score(Test_Y, predictions, pos_label=0)
        print(f"{model_name} Accuracy Score -> {accuracy:.2f}")
        print(f"{model_name} Precision Score -> {precision:.2f}")
        print(f"{model_name} Recall Score -> {recall:.2f}")
        print(f"{model_name} F1 Score -> {f1:.2f}")
        print("-" * 30)

    # Print metrics for each model
    print_metrics("Dummy Classifier", predictions_dummy)
    print_metrics("Naive Bayes", predictions_NB)
    print_metrics("SVM", predictions_SVM)
    print_metrics("Decision Tree", predictions_clf)

if __name__ == "__main__":
    main()