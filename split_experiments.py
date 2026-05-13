import pandas as pd

from preprocessing import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# Global settings
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Step 1: Load and clean dataset
# ---------------------------------------------------------

def load_dataset(csv_file):
    """
    Loads the CSV file and cleans the text column.

    The CSV file must contain:
    - text
    - emotion
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


# ---------------------------------------------------------
# Step 2: Split data into train and test
# ---------------------------------------------------------

def split_dataset(df, train_size):
    """
    Splits the dataset into training and testing sets.

    Example:
    train_size = 0.40 means:
    40% training data
    60% testing data

    stratify=y keeps class proportions similar in both sets.
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


# ---------------------------------------------------------
# Step 3: Convert text to TF-IDF features
# ---------------------------------------------------------

def create_tfidf_features(X_train_text, X_test_text):
    """
    Converts text into numerical TF-IDF features.

    Important:
    - fit_transform is used ONLY on the training data
    - transform is used on the testing data

    This avoids data leakage.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, vectorizer


# ---------------------------------------------------------
# Step 4: Define machine learning models
# ---------------------------------------------------------

def get_models():
    """
    Creates the four machine learning models required for the project.

    Models:
    1. Decision Tree
    2. SVM
    3. kNN
    4. Naive Bayes
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


# ---------------------------------------------------------
# Step 5: Train and evaluate one model
# ---------------------------------------------------------

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, split_name):
    """
    Trains one model and calculates evaluation scores.
    """

    print("Training model:", model_name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test labels
    y_pred = model.predict(X_test)

    # Calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    result = {
        "split": split_name,
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    return result


# ---------------------------------------------------------
# Step 6: Print results separately for each split
# ---------------------------------------------------------

def print_results_by_split(results_df):
    """
    Prints a clean table for each train/test split.

    The table is sorted from best to worst using:
    1. macro_f1
    2. accuracy

    The rank column fixes the confusing pandas index problem.
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


# ---------------------------------------------------------
# Step 7: Print overall best model
# ---------------------------------------------------------

def print_overall_best(results_df):
    """
    Prints one final table sorted across all splits.
    Also prints the best overall model.
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


# ---------------------------------------------------------
# Step 8: Main experiment function
# ---------------------------------------------------------

def run_split_experiments(csv_file):
    """
    Runs all split experiments.

    Tested splits:
    - 40% train / 60% test
    - 30% train / 70% test
    - 20% train / 80% test

    For each split, the code trains:
    - Decision Tree
    - SVM
    - kNN
    - Naive Bayes
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

    # Save raw results before printing
    results_df.to_csv("split_model_comparison_results.csv", index=False)

    # Print clean results
    print_results_by_split(results_df)

    # Print best overall result
    print_overall_best(results_df)

    print("\nSaved file:")
    print("split_model_comparison_results.csv")

    return results_df


# ---------------------------------------------------------
# Program starts here
# ---------------------------------------------------------

if __name__ == "__main__":
    run_split_experiments("train_emotion.csv")