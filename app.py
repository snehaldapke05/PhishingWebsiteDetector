from flask import Flask, request, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

# Train model
X = data["url"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)


def is_suspicious_url(url):
    """Catch common phishing wording that a tiny training set may miss."""
    normalized_url = url.strip().lower()
    suspicious_patterns = (
        r"verify.*(account|login|password)",
        r"(account|password|login).*(verify|confirm|update)",
        r"free[-_ ]?prize",
        r"winner|claim[-_ ]?now",
        r"bank[-_ ]?login",
        r"account[-_ ]?suspended",
    )
    return any(re.search(pattern, normalized_url) for pattern in suspicious_patterns)


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        url = request.form["url"]

        if is_suspicious_url(url):
            prediction = [1]
        else:
            url_vectorized = vectorizer.transform([url])
            prediction = model.predict(url_vectorized)

        if prediction[0] == 1:
            result = "⚠️ This website may be PHISHING!"
        else:
            result = "✅ This website appears SAFE."

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)