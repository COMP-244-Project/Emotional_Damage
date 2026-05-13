import pandas as pd

df = pd.read_csv("train_emotion.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df["emotion"].value_counts())

# Text length
df["text_length"] = df["text"].apply(len)

print("\nText length statistics:")
print(df["text_length"].describe())