import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_preprocess(csv_file):

    # Load dataset
    df = pd.read_csv(csv_file)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=5000
    )

    X = vectorizer.fit_transform(df["clean_text"])

    y = df["emotion"]

    return X, y, vectorizer


