import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
df = pd.concat([fake, real])

# Features and target
X = df["text"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save files
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model Saved Successfully")