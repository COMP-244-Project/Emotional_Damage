"""
analysis.py

Dataset analysis module for the emotion classification project.

This file is used to inspect the training dataset before model development.
It prints basic information about the dataset, including sample rows, dataset
shape, missing values, duplicate rows, emotion label distribution, and text
length statistics.

The purpose of this file is to support the analysis and report-writing part of
the project. It is not required for final model training or prediction.

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
import pandas as pd


def analyze_dataset(csv_file):
    """
    Analyze the structure and basic statistics of the dataset.

    This function loads the CSV dataset and prints useful information for
    understanding the data before training the machine learning models. It
    checks the dataset size, missing values, duplicate rows, emotion label
    distribution, and text length statistics.

    @param csv_file: Path to the CSV file containing the dataset.
    @type csv_file: str

    @return: None
    @rtype: None
    """

    # Load the dataset from the CSV file
    df = pd.read_csv(csv_file)

    # Print the first few rows to check the dataset format
    print("First rows:")
    print(df.head())

    # Print the number of rows and columns
    print("\nShape:")
    print(df.shape)

    # Check for missing values in each column
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check for duplicate rows in the dataset
    print("\nDuplicates:")
    print(df.duplicated().sum())

    # Show how many examples belong to each emotion label
    print("\nLabel distribution:")
    print(df["emotion"].value_counts())

    # Create a text length column for basic text statistics
    df["text_length"] = df["text"].apply(len)

    # Print descriptive statistics for text length
    print("\nText length statistics:")
    print(df["text_length"].describe())

# Run analysis only when this file is executed directly
if __name__ == "__main__":
    analyze_dataset("train_emotion.csv")