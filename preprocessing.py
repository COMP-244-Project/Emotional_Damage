import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    Cleans one text sample.

    Steps:
    1. Convert text to string
    2. Convert to lowercase
    3. Remove URLs
    4. Remove punctuation and numbers
    5. Remove extra spaces
    """

    # Convert to string in case input is not already text
    text = str(text)

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
    """
    Loads the CSV file, cleans the text column,
    applies TF-IDF, and returns X, y, and vectorizer.
    """

    # Load dataset
    df = pd.read_csv(csv_file)

    # Remove missing text/emotion rows
    df = df.dropna(subset=["text", "emotion"])

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )

    # X contains numerical text features
    X = vectorizer.fit_transform(df["clean_text"])

    # y contains emotion labels
    y = df["emotion"]

    return X, y, vectorizer