# Email Spam Detection using Machine Learning (Part 1)


# ============================================================
# EMAIL SPAM DETECTION USING MACHINE LEARNING
# PART 1 : DATA LOADING, EXPLORATION & PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# STEP 1 : IMPORT LIBRARIES
# ------------------------------------------------------------

# pandas -> Used for loading and manipulating datasets
import pandas as pd

# numpy -> Used for numerical operations
import numpy as np

# matplotlib -> Used for plotting graphs
import matplotlib.pyplot as plt

# seaborn -> Used for beautiful statistical visualizations
import seaborn as sns

# Regular Expressions
# Used for removing URLs, punctuation, numbers, etc.
import re

# String library
# Contains punctuation characters
import string

# Natural Language Toolkit
# Used for stopwords, tokenization and lemmatization
import nltk

# Download required NLTK resources (only needed once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------
# STEP 2 : LOAD DATASET
# ------------------------------------------------------------

# Load the spam dataset
df = pd.read_csv("spam.csv")

# Display first five rows
print(df.head())

# ------------------------------------------------------------
# STEP 3 : BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nColumn Names")
print(df.columns)

# ------------------------------------------------------------
# STEP 4 : REMOVE UNNECESSARY COLUMNS
# ------------------------------------------------------------

# Many spam datasets contain empty columns such as:
# Unnamed: 2
# Unnamed: 3
# Unnamed: 4

# Remove columns containing only missing values
df = df.dropna(axis=1)

print("\nRemaining Columns")
print(df.columns)

# ------------------------------------------------------------
# STEP 5 : RENAME COLUMNS
# ------------------------------------------------------------

# Original columns are generally:
# v1 -> Spam/Ham
# v2 -> Message

df.columns = ["label", "message"]

print(df.head())

# ------------------------------------------------------------
# STEP 6 : CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values")

print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 7 : CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\nDuplicate Rows")

print(df.duplicated().sum())

# ------------------------------------------------------------
# STEP 8 : REMOVE DUPLICATES
# ------------------------------------------------------------

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")

print(df.shape)

# ------------------------------------------------------------
# STEP 9 : LABEL ENCODING
# ------------------------------------------------------------

# ham -> 0
# spam -> 1

df["label"] = df["label"].map({
    "ham":0,
    "spam":1
})

print(df.head())

# ------------------------------------------------------------
# STEP 10 : DATASET DISTRIBUTION
# ------------------------------------------------------------

print("\nLabel Counts")

print(df["label"].value_counts())

# ------------------------------------------------------------
# STEP 11 : VISUALIZE CLASS DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x=df["label"]
)

plt.title("Spam vs Ham Distribution")

plt.xlabel("Label")

plt.ylabel("Count")

plt.show()

# ------------------------------------------------------------
# STEP 12 : FEATURE ENGINEERING
# ------------------------------------------------------------

# Character Count

df["char_count"] = df["message"].apply(len)

# Word Count

df["word_count"] = df["message"].apply(
    lambda x: len(x.split())
)

print(df.head())

# ------------------------------------------------------------
# STEP 13 : VISUALIZE CHARACTER COUNT
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="char_count",
    hue="label",
    bins=50
)

plt.title("Character Count Distribution")

plt.show()

# ------------------------------------------------------------
# STEP 14 : VISUALIZE WORD COUNT
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="word_count",
    hue="label",
    bins=50
)

plt.title("Word Count Distribution")

plt.show()

# ------------------------------------------------------------
# STEP 15 : CREATE STOPWORDS
# ------------------------------------------------------------

# Stopwords are common words that usually
# don't help in spam classification.

stop_words = set(stopwords.words("english"))

# ------------------------------------------------------------
# STEP 16 : CREATE LEMMATIZER
# ------------------------------------------------------------

lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------
# STEP 17 : TEXT PREPROCESSING FUNCTION
# ------------------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\\S+@\\S+", "", text)

    # Remove numbers
    text = re.sub(r"\\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Split into words
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in stop_words

    ]

    # Join words back into a sentence
    return " ".join(words)

# ------------------------------------------------------------
# STEP 18 : APPLY PREPROCESSING
# ------------------------------------------------------------

df["processed_message"] = df["message"].apply(
    preprocess_text
)

# ------------------------------------------------------------
# STEP 19 : DISPLAY ORIGINAL VS PROCESSED
# ------------------------------------------------------------

print("\nOriginal Message\n")

print(df["message"][0])

print("\nProcessed Message\n")

print(df["processed_message"][0])

# ------------------------------------------------------------
# STEP 20 : FINAL DATASET
# ------------------------------------------------------------

print(df.head())

print("\nFinal Dataset Shape")

print(df.shape)

# ------------------------------------------------------------
# END OF PART 1
# ------------------------------------------------------------

# Email Spam Detection using Machine Learning (Part 2)


# ============================================================
# EMAIL SPAM DETECTION USING MACHINE LEARNING
# PART 2 : MODEL BUILDING, TRAINING & EVALUATION
# ============================================================

# ------------------------------------------------------------
# STEP 21 : IMPORT MACHINE LEARNING LIBRARIES
# ------------------------------------------------------------

# TF-IDF converts text into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Splits dataset into training and testing data.
from sklearn.model_selection import train_test_split

# Naive Bayes classifier for text classification.
from sklearn.naive_bayes import MultinomialNB

# Evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Save trained model.
import joblib

# ------------------------------------------------------------
# STEP 22 : CREATE FEATURES AND LABELS
# ------------------------------------------------------------

# Input Feature
X = df["processed_message"]

# Target
y = df["label"]

# ------------------------------------------------------------
# STEP 23 : TF-IDF VECTORIZATION
# ------------------------------------------------------------

# Convert text into numerical vectors.

tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(X)

print("TF-IDF Shape")
print(X.shape)

# ------------------------------------------------------------
# STEP 24 : TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("\nTraining Samples:", X_train.shape[0])

print("Testing Samples:", X_test.shape[0])

# ------------------------------------------------------------
# STEP 25 : BUILD MODEL
# ------------------------------------------------------------

model = MultinomialNB()

print(model)

# ------------------------------------------------------------
# STEP 26 : TRAIN MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

# ------------------------------------------------------------
# STEP 27 : MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# STEP 28 : MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy")

print(round(accuracy*100,2), "%")

# ------------------------------------------------------------
# STEP 29 : PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred
)

print("\nPrecision")

print(round(precision*100,2), "%")

# ------------------------------------------------------------
# STEP 30 : RECALL
# ------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred
)

print("\nRecall")

print(round(recall*100,2), "%")

# ------------------------------------------------------------
# STEP 31 : F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred
)

print("\nF1 Score")

print(round(f1*100,2), "%")

# ------------------------------------------------------------
# STEP 32 : CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ------------------------------------------------------------
# STEP 33 : CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)

# ------------------------------------------------------------
# STEP 34 : VISUALIZE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ------------------------------------------------------------
# STEP 35 : TEST CUSTOM EMAILS
# ------------------------------------------------------------

emails = [

    "Congratulations! You won a free iPhone. Click here now.",

    "Meeting has been postponed to tomorrow morning.",

    "Claim your prize immediately by visiting this website.",

    "Please submit your assignment before 5 PM."

]

emails = [

    preprocess_text(email)

    for email in emails

]

emails = tfidf.transform(emails)

predictions = model.predict(emails)

print("\nCustom Predictions\n")

for email, prediction in zip(emails, predictions):

    if prediction == 1:

        print("Spam")

    else:

        print("Ham")

# ------------------------------------------------------------
# STEP 36 : USER INPUT PREDICTION
# ------------------------------------------------------------

user_email = input(

    "\nEnter an Email : "

)

processed_email = preprocess_text(

    user_email

)

vector = tfidf.transform(

    [processed_email]

)

prediction = model.predict(

    vector

)

print()

if prediction[0] == 1:

    print("Prediction : Spam Email")

else:

    print("Prediction : Not Spam")

# ------------------------------------------------------------
# STEP 37 : SAVE MODEL
# ------------------------------------------------------------

joblib.dump(

    model,

    "spam_model.pkl"

)

print("\nModel Saved Successfully")

# ------------------------------------------------------------
# STEP 38 : SAVE TF-IDF VECTORIZER
# ------------------------------------------------------------

joblib.dump(

    tfidf,

    "tfidf_vectorizer.pkl"

)

print("TF-IDF Vectorizer Saved Successfully")

# ------------------------------------------------------------
# STEP 39 : LOAD MODEL
# ------------------------------------------------------------

loaded_model = joblib.load(

    "spam_model.pkl"

)

loaded_vectorizer = joblib.load(

    "tfidf_vectorizer.pkl"

)

print("\nModel Loaded Successfully")

# ------------------------------------------------------------
# STEP 40 : VERIFY SAVED MODEL
# ------------------------------------------------------------

sample = "Congratulations! You have won 50000 rupees."

sample = preprocess_text(sample)

sample = loaded_vectorizer.transform([sample])

prediction = loaded_model.predict(sample)

print()

if prediction[0] == 1:

    print("Loaded Model Prediction : Spam")

else:

    print("Loaded Model Prediction : Ham")

# ============================================================
# END OF PROJECT
# ============================================================

