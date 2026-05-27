# Emotional Damage — Emotion Classification Project

A machine learning project for multi-class emotion classification from short text messages using classical NLP techniques and supervised learning models.

---

## Project Overview

This project was developed for **COMP-244** and investigates how different classical machine learning algorithms perform on an emotion classification task using TF-IDF text representations.

The system classifies text into one of seven emotion categories:

* sadness
* neutral
* worry
* happiness
* fun
* surprise
* hate

The project focuses on:

* text preprocessing,
* feature engineering with TF-IDF,
* comparative evaluation of multiple classifiers,
* handling class imbalance,
* analysing performance under limited training data.

---

# Repository Structure

```
Emotional_Damage/
│
├── analysis.py
├── preprocessing.py
├── split_experiments.py
├── main.py
│
├── train_emotion.csv
│
├── split_label_distributions.csv
├── split_model_comparison_results.csv
├── main_model_comparison_results.csv
│
└── README.md
```

---

# File Descriptions

## `analysis.py`

Performs exploratory data analysis (EDA) on the dataset.

### Responsibilities

* Loads dataset
* Checks:

  * missing values
  * duplicate samples
  * label distribution
  * dataset shape
* Computes text length statistics

This file was used to better understand dataset characteristics before modelling.

---

## `preprocessing.py`

Contains all preprocessing and feature extraction logic.

### Preprocessing Pipeline

The text cleaning process includes:

* lowercasing
* URL removal
* punctuation removal
* number removal
* whitespace normalization

### TF-IDF Vectorization

The project uses `TfidfVectorizer` with tuned parameters:

```python
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=2
)
```

### Why These Parameters?

| Parameter              | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| `stop_words='english'` | Removes common non-informative words                |
| `ngram_range=(1,2)`    | Captures both single words and short phrases        |
| `max_features=10000`   | Limits vocabulary size and reduces noise            |
| `sublinear_tf=True`    | Dampens impact of extremely frequent words          |
| `min_df=2`             | Removes extremely rare words unlikely to generalize |

The vectorizer is always fitted **only on training data** to prevent data leakage.

---

## `split_experiments.py`

Runs experiments using multiple train/test split configurations.

### Evaluated Splits

* 40% train / 60% test
* 30% train / 70% test
* 20% train / 80% test

### Purpose

This file evaluates how model performance changes when less training data is available.

### Metrics Computed

* Accuracy
* Macro F1
* Weighted F1

Results are exported to:

* `split_model_comparison_results.csv`
* `split_label_distributions.csv`

---

## `main.py`

Main training and evaluation pipeline.

### Models Evaluated

The project compares four classical machine learning algorithms:

| Model                   | Description                                  |
| ----------------------- | -------------------------------------------- |
| Decision Tree           | Tree-based classifier with constrained depth |
| LinearSVC               | Linear Support Vector Machine                |
| kNN                     | k-Nearest Neighbours using cosine similarity |
| Multinomial Naive Bayes | Probabilistic classifier for sparse text     |

### Ensemble Experiments

The project also explored ensemble approaches during experimentation.

Voting-based ensembles were tested conceptually to combine predictions from multiple classifiers. The goal was to improve robustness by aggregating predictions from different learning paradigms.

However:

* LinearSVC consistently outperformed ensemble combinations,
* ensemble models added computational complexity,
* macro F1 improvements were minimal.

As a result, the final system retained the simpler and more efficient LinearSVC pipeline.

### Final Selected Model

The final model is:

```
TF-IDF Vectorizer + LinearSVC
```

### Why LinearSVC?

The final model uses `LinearSVC`, which is a **linear Support Vector Machine**.

It does **not** apply the kernel trick.

This design choice is appropriate because TF-IDF generates:

* extremely high-dimensional
* sparse feature vectors

In such spaces, text classes are often approximately linearly separable, meaning linear classifiers perform very effectively while remaining computationally efficient.

Compared with kernel SVMs:

* `LinearSVC` scales significantly better,
* training is faster,
* memory usage is lower,
* sparse matrices are handled efficiently.

Balanced class weights were also used to improve performance on underrepresented classes such as `surprise`.

---

# Machine Learning Discussion

## Why Decision Trees Performed Poorly

Decision Trees tend to overfit high-dimensional sparse TF-IDF spaces because they attempt to split on thousands of individual features.

This leads to:

* unstable decision boundaries,
* memorization of training noise,
* weak generalization.

---

## Why kNN Was Limited

Although cosine similarity improves kNN performance for text data, kNN still suffers from the curse of dimensionality.

In sparse TF-IDF spaces:

* distances between samples become less informative,
* neighborhood quality degrades,
* classification becomes less reliable.

---

## Why Naive Bayes Worked Well

Despite its unrealistic independence assumption, Naive Bayes performs surprisingly well on many NLP tasks because:

* word frequencies remain highly informative,
* probabilistic estimation is efficient,
* sparse text features align well with multinomial modelling.

---

## Why LinearSVC Performed Best

Linear SVMs are especially effective for text classification because:

* TF-IDF spaces are high-dimensional,
* many classes become approximately linearly separable,
* margin maximization improves generalization,
* sparse matrices are processed efficiently.

The model achieved the strongest macro F1 scores across all evaluated configurations.

---

# Evaluation Metrics

## Accuracy

Measures overall classification correctness.

```
Accuracy = Correct Predictions / Total Predictions
```

Useful as a general indicator but less reliable for imbalanced datasets.

---

## Macro F1

Averages F1 scores equally across all classes.

This metric is especially important because:

* the dataset is imbalanced,
* minority classes must still be evaluated fairly.

Macro F1 was treated as the primary evaluation metric.

---

## Weighted F1

Weights class F1 scores according to class frequency.

Provides a balance between:

* overall performance,
* class-sensitive evaluation.

---

# Dataset Notes

The dataset contains:

* 8000 labelled text samples
* 7 emotion categories

The dataset was checked for:

* missing values
* duplicate entries

The `surprise` class is underrepresented, which motivated the use of:

* macro F1 evaluation,
* balanced class weights.

---

# Final Conclusions

The experiments demonstrate that:

* TF-IDF remains highly effective for classical NLP pipelines,
* LinearSVC is extremely strong for sparse text classification,
* macro F1 is essential for imbalanced emotion datasets,
* simpler linear models can outperform more complex alternatives,
* ensemble methods did not provide sufficient improvement to justify added complexity.

The final system achieved the best balance between:

* accuracy,
* generalization,
* computational efficiency.

---

# Future Improvements

Potential future work includes:

* transformer-based models (BERT, RoBERTa),
* contextual embeddings,
* deep learning architectures,
* hyperparameter optimization,
* advanced ensemble methods,
* data augmentation for minority classes.

---

# Contributors

* Polina Marinic
* Andrei Iatsina
* Gleb Gaivoronskii

---

# Technologies Used

* Python
* scikit-learn
* pandas
* NumPy
* TF-IDF
* LinearSVC
* Classical Machine Learning
* NLP preprocessing
