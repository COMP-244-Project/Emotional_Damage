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
    Trains one model and prints evaluation results.
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
    Runs Decision Tree, SVM, kNN, and Naive Bayes experiments.
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
    results_df = results_df.sort_values(by="macro_f1", ascending=False)

    print("\n==============================")
    print("Final Model Comparison Table")
    print("==============================")
    print(results_df)

    return results_df


def train_test(train_csv, test_csv):
    """
    Final function required by the assignment.

    It trains the best model on the full training set,
    then predicts emotions for the test set.
    """

    # Load and preprocess training data
    X_train, y_train, vectorizer = load_and_preprocess(train_csv)

    # Train final selected model: SVM
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

    # Important:
    # Use transform, not fit_transform.
    # We must use the same vectorizer learned from the training data.
    X_test = vectorizer.transform(test_df["clean_text"])

    # Predict test labels
    predictions = model.predict(X_test)

    # Add predictions to output dataframe
    output = test_df.copy()
    output["emotion"] = predictions

    return output


if __name__ == "__main__":
    run_experiments("train_emotion.csv")