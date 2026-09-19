import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Input and output
X = data["url"]
y = data["label"]

# Convert URLs into numerical features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Model trained successfully!")

# Test a URL
url = input("Enter a website URL: ")

url_vectorized = vectorizer.transform([url])
prediction = model.predict(url_vectorized)

if prediction[0] == 1:
    print("⚠️ This website may be PHISHING!")
else:

    print("✅ This website appears SAFE.")
