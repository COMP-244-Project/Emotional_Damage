"""
preprocessing.py

Text preprocessing and feature extraction module for the emotion classification project.

This file contains the preprocessing functions used before training the machine
learning models. It cleans raw text data and converts the cleaned text into
TF-IDF feature vectors that can be used by scikit-learn classifiers.

The main preprocessing steps include lowercasing, removing URLs, removing
punctuation and numbers, removing extra spaces, and applying TF-IDF vectorization.

Author:
    Gleb Gaivoronskii, Andrey Iatsina, Polina Marinic

Version:
    1.0

Date:
    2026-05-27

Bugs:
    None known.

Copyright:
    University of Nicosia
"""
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    Clean one text message before feature extraction.

    This function prepares raw text for the emotion classification models. It
    converts the text to lowercase, removes URLs, removes punctuation and
    numbers, and replaces multiple spaces with a single space.

    @param text: Raw text message.
    @type text: str

    @return: Cleaned text message.
    @rtype: str
    """

    # Convert the text to lowercase
    text = text.lower()

    # Remove URLs from the text
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation, numbers, and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces from the cleaned text
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_preprocess(csv_file):
    """
    Load the dataset, clean the text, and create TF-IDF features.

    This function reads the training CSV file, applies text cleaning to the
    text column, and converts the cleaned text into numerical TF-IDF features.
    It also returns the target emotion labels and the fitted vectorizer.

    The vectorizer is returned so it can later be reused to transform test data
    with the same vocabulary learned from the training data.

    @param csv_file: Path to the CSV file containing the training dataset.
    @type csv_file: str

    @return: TF-IDF feature matrix, emotion labels, and fitted TF-IDF vectorizer.
    @rtype: tuple
    """

    # Load the dataset from the CSV file
    df = pd.read_csv(csv_file)

    # Create a cleaned version of the text column
    df["clean_text"] = df["text"].apply(clean_text)

    # Create the TF-IDF vectorizer used for feature extraction
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )

    # Fit the vectorizer on the cleaned text and create the feature matrix
    X = vectorizer.fit_transform(df["clean_text"])

    # Store the emotion column as the target labels
    y = df["emotion"]

    return X, y, vectorizer