"""
split_experiments.py

Experiment module for testing different train/validation splits.

This file contains the code used to evaluate several machine learning models
on different training and validation split sizes. The purpose of this script is
to check whether model performance changes when the amount of training data is
increased or decreased.

The tested models include Decision Tree, Support Vector Machine,
k-Nearest Neighbors, and Naive Bayes. Each model is trained and evaluated on
multiple split settings, and the results are compared using accuracy,
macro F1-score, and weighted F1-score.

The final comparison table helps identify which model and split combination
performs best for the emotion classification task.

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

from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score


# Global settings
RANDOM_STATE = 42

LABELS = [
    "sadness",
    "neutral",
    "worry",
    "happiness",
    "fun",
    "surprise",
    "hate"
]

def load_dataset(csv_file):
    """
    Load the dataset and prepare the text column for experiments.

    This function reads the CSV file, checks that the required columns are
    available, removes rows with missing values, converts the text column to
    string format, and creates a cleaned text column using the preprocessing
    function.

    @param csv_file: Path to the CSV dataset file.
    @type csv_file: str

    @return: Dataframe containing the original data and the cleaned text column.
    @rtype: pandas.DataFrame
    """

    print("\nLoading dataset...")

    df = pd.read_csv(csv_file)

    # Check that required columns exist
    if "text" not in df.columns:
        raise ValueError("The CSV file must contain a 'text' column.")

    if "emotion" not in df.columns:
        raise ValueError("The CSV file must contain an 'emotion' column.")

    # Remove rows with missing text or emotion
    df = df.dropna(subset=["text", "emotion"])

    # Make sure text is string type
    df["text"] = df["text"].astype(str)

    # Clean text using clean_text from preprocessing.py
    df["clean_text"] = df["text"].apply(clean_text)

    print("Dataset loaded successfully.")
    print("Total samples:", len(df))

    return df

def split_dataset(df, train_size):
    """
    Split the dataset into training and testing parts.

    The train_size value controls how much of the dataset is used for training.
    For example, train_size = 0.40 means that 40% of the data is used for
    training and 60% is used for testing.

    Stratified splitting is used so that the emotion labels keep similar
    proportions in both the training and testing sets.

    @param df: Dataset containing the cleaned text and emotion labels.
    @type df: pandas.DataFrame

    @param train_size: Proportion of the dataset used for training.
    @type train_size: float

    @return: Training text, testing text, training labels, and testing labels.
    @rtype: tuple
    """

    X = df["clean_text"]
    y = df["emotion"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train_text, X_test_text, y_train, y_test

def create_tfidf_features(X_train_text, X_test_text):
    """
    Convert cleaned text into TF-IDF feature matrices.

    The TF-IDF vectorizer is fitted only on the training text. The testing text
    is transformed using the same fitted vectorizer. This avoids data leakage
    and keeps the feature space consistent between training and testing data.

    @param X_train_text: Cleaned training text.
    @type X_train_text: pandas Series

    @param X_test_text: Cleaned testing text.
    @type X_test_text: pandas Series

    @return: Training feature matrix, testing feature matrix, and fitted
             TF-IDF vectorizer.
    @rtype: tuple
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, vectorizer

def get_models():
    """
    Create the machine learning models used in the experiments.

    The project compares four classifiers: Decision Tree, Support Vector
    Machine, k-Nearest Neighbors, and Naive Bayes. The model settings are based
    on the selected experiment configuration for this project.

    @return: Dictionary where each key is a model name and each value is a
             scikit-learn model object.
    @rtype: dict
    """

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=40,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "SVM": LinearSVC(
            C=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "kNN": KNeighborsClassifier(
            n_neighbors=7,
            metric="cosine"
        ),

        "Naive Bayes": MultinomialNB(
            alpha=0.1
        )
    }

    return models

def evaluate_model(model_name, model, X_train, X_val, y_train, y_val, split_name):
    """
    Train and evaluate one model for a specific data split.

    This function trains the given model on the training data, predicts emotion
    labels for the validation/testing data, calculates the main evaluation
    scores, and returns the results together with the split name.

    @param model_name: Name of the model being evaluated.
    @type model_name: str

    @param model: Machine learning model with fit() and predict() methods.
    @type model: object

    @param X_train: Training feature matrix.
    @type X_train: scipy sparse matrix

    @param X_val: Validation or testing feature matrix.
    @type X_val: scipy sparse matrix

    @param y_train: Training labels.
    @type y_train: pandas Series

    @param y_val: Validation or testing labels.
    @type y_val: pandas Series

    @param split_name: Description of the train/test split.
    @type split_name: str

    @return: Dictionary containing the split name, model name, accuracy,
             macro F1-score, and weighted F1-score.
    @rtype: dict
    """

    print("Training model:", model_name)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict emotion labels for the validation/testing data
    y_pred = model.predict(X_val)

    # Calculate evaluation scores
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    result = {
        "split": split_name,
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    return result

def print_results_by_split(results_df):
    """
    Print model results separately for each train/test split.

    For each split, the models are sorted from best to worst using macro
    F1-score first and accuracy second. A clear rank column is added so the
    output is easier to read than the default pandas index.

    @param results_df: Dataframe containing all model evaluation results.
    @type results_df: pandas.DataFrame

    @return: None
    @rtype: None
    """

    print("\n\n" + "=" * 80)
    print("RESULTS SEPARATELY FOR EACH TRAIN / TEST SPLIT")
    print("=" * 80)

    split_names = results_df["split"].unique()

    for split_name in split_names:
        split_table = results_df[results_df["split"] == split_name].copy()

        # Sort best to worst
        split_table = split_table.sort_values(
            by=["macro_f1", "accuracy"],
            ascending=False
        )

        # Reset old pandas index
        split_table = split_table.reset_index(drop=True)

        # Add clear ranking column
        split_table.insert(0, "rank", split_table.index + 1)

        print("\n" + "-" * 80)
        print(split_name)
        print("-" * 80)

        print(
            split_table[
                [
                    "rank",
                    "model",
                    "accuracy",
                    "macro_f1",
                    "weighted_f1"
                ]
            ].to_string(index=False)
        )

        # Print best model for this split
        best_in_split = split_table.iloc[0]

        print("\nBest model for this split:")
        print("Model:", best_in_split["model"])
        print("Accuracy:", round(best_in_split["accuracy"], 4))
        print("Macro F1:", round(best_in_split["macro_f1"], 4))
        print("Weighted F1:", round(best_in_split["weighted_f1"], 4))

def print_overall_best(results_df):
    """
    Print the overall ranking across all splits.

    This function sorts all experiment results from best to worst using macro
    F1-score and accuracy. It then prints the full ranking table and highlights
    the best overall model and split combination.

    @param results_df: Dataframe containing all model evaluation results.
    @type results_df: pandas.DataFrame

    @return: None
    @rtype: None
    """

    overall_table = results_df.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False
    ).reset_index(drop=True)

    overall_table.insert(0, "overall_rank", overall_table.index + 1)

    print("\n\n" + "=" * 80)
    print("OVERALL BEST MODELS ACROSS ALL SPLITS")
    print("=" * 80)

    print(
        overall_table[
            [
                "overall_rank",
                "split",
                "model",
                "accuracy",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(index=False)
    )

    best_result = overall_table.iloc[0]

    print("\n" + "=" * 80)
    print("BEST OVERALL RESULT")
    print("=" * 80)

    print("Split:", best_result["split"])
    print("Model:", best_result["model"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("Macro F1:", round(best_result["macro_f1"], 4))
    print("Weighted F1:", round(best_result["weighted_f1"], 4))

def run_split_experiments(csv_file):
    """
    Run all train/test split experiments.

    This function loads the dataset, creates several train/test splits, converts
    text into TF-IDF features, trains each machine learning model, and stores
    the evaluation results.

    The tested splits are:
    - 40% train / 60% test
    - 30% train / 70% test
    - 20% train / 80% test

    The final results are saved to split_model_comparison_results.csv.

    @param csv_file: Path to the training CSV file.
    @type csv_file: str

    @return: Dataframe containing the results for all models and splits.
    @rtype: pandas.DataFrame
    """

    df = load_dataset(csv_file)

    split_settings = [
        ("40% train / 60% test", 0.40),
        ("30% train / 70% test", 0.30),
        ("20% train / 80% test", 0.20)
    ]

    all_results = []

    for split_name, train_size in split_settings:

        print("\n\n" + "#" * 80)
        print("Running experiment:", split_name)
        print("#" * 80)

        # Split dataset
        X_train_text, X_test_text, y_train, y_test = split_dataset(
            df,
            train_size
        )

        print("Training samples:", len(y_train))
        print("Testing samples:", len(y_test))

        # Convert text to TF-IDF features
        X_train, X_test, vectorizer = create_tfidf_features(
            X_train_text,
            X_test_text
        )

        # Get models
        models = get_models()

        # Train and evaluate each model
        for model_name, model in models.items():
            result = evaluate_model(
                model_name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                split_name
            )

            all_results.append(result)

    # Convert all results to dataframe
    results_df = pd.DataFrame(all_results)

    # Save all experiment results before printing the formatted tables
    results_df.to_csv("split_model_comparison_results.csv", index=False)

    # Print clean results
    print_results_by_split(results_df)

    # Print best overall result
    print_overall_best(results_df)

    print("\nSaved file:")
    print("split_model_comparison_results.csv")

    return results_df



# Run experiments only when this file is executed directly
if __name__ == "__main__":
    run_split_experiments("train_emotion.csv")