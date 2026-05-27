"""
main.py

Main training and evaluation module for the emotion classification project.

This file contains the main code used to train, evaluate, and compare several
machine learning models for emotion classification. The tested models are
Decision Tree, Support Vector Machine, k-Nearest Neighbors, and Naive Bayes.

The program loads the training dataset, applies preprocessing and TF-IDF
feature extraction, evaluates each model on a validation set, ranks the models
by performance, and saves the comparison results to a CSV file.

The final train_test function trains the selected model on the full training
dataset and predicts emotion labels for a separate test dataset.

Author:
    Gleb Gaivoronskii, Andrey Iatsina, Polina Marinic

Version:
    1.0

Date:
    2026-05-26

Bugs:
    None known.

Copyright:
    University of Nicosia
"""
import pandas as pd

from preprocessing import clean_text, load_and_preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


LABELS = [
    "sadness",
    "neutral",
    "worry",
    "happiness",
    "fun",
    "surprise",
    "hate"
]

def evaluate_model(model_name, model, X_train, X_val, y_train, y_val):
    """
    Train and evaluate one machine learning model.

    This helper function trains the given model on the training data, predicts
    labels for the validation data, calculates the main evaluation scores, and
    prints a classification report.

    @param model_name: Name of the model being evaluated.
    @type model_name: str

    @param model: Machine learning model with fit() and predict() methods.
    @type model: object

    @param X_train: Training feature matrix.
    @type X_train: scipy sparse matrix

    @param X_val: Validation feature matrix.
    @type X_val: scipy sparse matrix

    @param y_train: Training labels.
    @type y_train: pandas Series

    @param y_val: Validation labels.
    @type y_val: pandas Series

    @return: Dictionary containing the model name, accuracy, macro F1-score,
             and weighted F1-score.
    @rtype: dict
    """

    print("\n==============================")
    print("Training model:", model_name)
    print("==============================")

    # Train the model
    model.fit(X_train, y_train)

    # Predict validation labels
    y_pred = model.predict(X_val)

    # Calculate scores
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    print("\nClassification report:")
    print(classification_report(y_val, y_pred, labels=LABELS))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def run_experiments(train_csv):
    """
    Run all model evaluation experiments.

    This function loads and preprocesses the training dataset, splits it into
    training and validation sets, trains several machine learning models, and
    compares their performance. The final results are sorted from best to worst
    using macro F1-score and accuracy.

    The ranked comparison table is printed to the console and saved as a CSV
    file named main_model_comparison_results.csv.

    @param train_csv: Path to the training CSV file.
    @type train_csv: str

    @return: Ranked table containing the evaluation results for all models.
    @rtype: pandas.DataFrame
    """

    # Load and preprocess full dataset
    X, y, vectorizer = load_and_preprocess(train_csv)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Define models
    models = [
        (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=40,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42
            )
        ),
        (
            "SVM",
            LinearSVC(
                C=0.5,
                class_weight="balanced",
                random_state=42
            )
        ),
        (
            "kNN",
            KNeighborsClassifier(
                n_neighbors=7,
                metric="cosine"
            )
        ),
        (
            "Naive Bayes",
            MultinomialNB(
                alpha=0.1
            )
        )
    ]

    results = []

    for model_name, model in models:
        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_val,
            y_train,
            y_val
        )
        results.append(result)

    # Create comparison table
    results_df = pd.DataFrame(results)

    # Sort from best to worst
    results_df = results_df.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False
    )

    # Remove old pandas index after sorting
    results_df = results_df.reset_index(drop=True)

    # Add real ranking column
    results_df.insert(0, "rank", results_df.index + 1)

    print("\n==============================")
    print("Final Model Comparison Table")
    print("==============================")

    print(
        results_df[
            [
                "rank",
                "model",
                "accuracy",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(index=False)
    )

    # Save clean ranked table
    results_df.to_csv("main_model_comparison_results.csv", index=False)

    # Print best model clearly
    best_model = results_df.iloc[0]

    print("\nBest model:")
    print("Rank:", best_model["rank"])
    print("Model:", best_model["model"])
    print("Accuracy:", round(best_model["accuracy"], 4))
    print("Macro F1:", round(best_model["macro_f1"], 4))
    print("Weighted F1:", round(best_model["weighted_f1"], 4))

    print("\nSaved file:")
    print("main_model_comparison_results.csv")

    return results_df


def train_test(train_csv, test_csv):
    """
    Train the final selected model and predict labels for the test dataset.

    This is the final function required for the assignment. It trains the
    selected best-performing model on the full training dataset, applies the
    same preprocessing and TF-IDF vectorizer to the test dataset, and returns
    the test data with predicted emotion labels.

    The same vectorizer fitted on the training data must be used to transform
    the test data. This avoids learning new features from the test set.

    @param train_csv: Path to the training CSV file.
    @type train_csv: str

    @param test_csv: Path to the test CSV file.
    @type test_csv: str

    @return: Test dataset with an additional column containing predicted
             emotion labels.
    @rtype: pandas.DataFrame
    """

    # Load and preprocess training data
    X_train, y_train, vectorizer = load_and_preprocess(train_csv)

    # Final tuned SVM model
    model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Load test data
    test_df = pd.read_csv(test_csv)

    # Clean test text using the same clean_text function
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    # Use the same vectorizer that was fitted on the training data.
    # This keeps the test features consistent with the training features.
    X_test = vectorizer.transform(test_df["clean_text"])

    # Predict emotion labels for the test data
    predictions = model.predict(X_test)

    # Create a copy of the test dataframe and add the predicted labels
    output = test_df.copy()
    output["emotion"] = predictions

    return output


if __name__ == "__main__":
    # Run experiments only when this file is executed directly.
    # This block will not run if main.py is imported by another file.
    run_experiments("train_emotion.csv")