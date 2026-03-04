import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data (only first time needed)
nltk.download('stopwords')
nltk.download('wordnet')

# Dataset
data = {
    "review": [
        "I love this product",
        "This is amazing",
        "Excellent quality",
        "Very happy with this",
        "Best purchase ever",
        "Super fast delivery",
        "Worst product",
        "Very bad experience",
        "I hate this",
        "Waste of money",
        "Totally disappointed",
        "Not worth buying",
        "Fantastic service",
        "Highly recommended",
        "Very satisfied",
        "Poor quality",
        "Terrible support",
        "Awful experience",
        "Great value",
        "Absolutely wonderful"
    ],
    "sentiment": [
        "Positive","Positive","Positive","Positive","Positive",
        "Positive","Negative","Negative","Negative","Negative",
        "Negative","Negative","Positive","Positive","Positive",
        "Negative","Negative","Negative","Positive","Positive"
    ]
}

df = pd.DataFrame(data)

# Text Cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_review"] = df["review"].apply(clean_text)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# Stratified split (IMPORTANT FIX)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

# Manual testing
while True:
    new_review = input("\nEnter a review (or type 'exit' to stop): ")

    if new_review.lower() == "exit":
        print("Program Ended.")
        break

    cleaned = clean_text(new_review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    print("Predicted Sentiment:", prediction[0])
