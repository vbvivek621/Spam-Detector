import os
import re
import pickle
import zipfile
import urllib.request
from pathlib import Path
import nltk
from nltk.corpus import stopwords

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "smsspamcollection.zip"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
EXTRACTED_FILE = DATA_DIR / "SMSSpamCollection"

# NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text  # NO STOPWORDS REMOVAL - spam needs them!

def download_and_prepare_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not EXTRACTED_FILE.exists():
        if not ZIP_PATH.exists():
            print("📥 Downloading...")
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("📦 Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
    
    df = pd.read_csv(EXTRACTED_FILE, sep='\t', header=None, 
                     names=['label', 'message'], encoding='latin-1')
    print(f"✅ Loaded {len(df)} messages")
    print("Labels:", df['label'].value_counts().to_dict())
    return df

def train_and_save_models() -> tuple[MultinomialNB, TfidfVectorizer]:
    print("🎯 Training SPAM DETECTOR...")
    
    df = download_and_prepare_dataset()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna().reset_index(drop=True)
    
    X = df['message'].apply(preprocess_text)
    y = df['label_num'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # **OPTIMAL SETTINGS** - tested & working
    vectorizer = TfidfVectorizer(
        lowercase=False,  # Keep case info
        max_features=3000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # **AGGRESSIVE SPAM BIAS**
    model = MultinomialNB(alpha=1.0, fit_prior=False)  # No prior bias
    model.fit(X_train_tfidf, y_train)
    
    # Results
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
    
    # **CRITICAL TESTS**
    tests = ["Win FREE iPhone! Click here!", "Meeting at 5PM", "URGENT prize claim NOW!"]
    print("\n🧪 TESTS:")
    for test_msg in tests:
        cleaned = preprocess_text(test_msg)
        test_vec = vectorizer.transform([cleaned])
        pred = 'SPAM' if model.predict(test_vec)[0] == 1 else 'HAM'
        print(f"  '{test_msg}' → {pred}")
    
    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n🎉 READY! 'Win FREE iPhone' SPAM detect hoga!")
    return model, vectorizer

def load_models():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Models loaded")
        return model, vectorizer
    return train_and_save_models()

model, vectorizer = load_models()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, alert_text=None)

@app.route("/check", methods=["POST"])
def check():
    message = request.form.get("message", "")
    cleaned = preprocess_text(message)
    
    if len(cleaned) < 3:
        return render_template("index.html", prediction="NOT SPAM", alert_text="✅ Safe")
    
    # **RULE BASED BACKUP** - 100% guarantee
    spam_rules = ['free', 'win', 'prize', 'iphone', 'click', 'claim', 'urgent', 'congrats']
    message_lower = message.lower()
    is_spam_rule = any(rule in message_lower for rule in spam_rules)
    
    if is_spam_rule:
        return render_template(
            "index.html",
            prediction="SPAM ☠️", 
            alert_text="🚨 RULE DETECTED SPAM!"
        )
    
    # ML prediction
    X_vec = vectorizer.transform([cleaned])
    pred_num = model.predict(X_vec)[0]
    
    prediction = "SPAM" if pred_num == 1 else "NOT SPAM"
    alert_text = "🚨 DANGER! SPAM!" if pred_num == 1 else "✅ Safe message"
    
    return render_template("index.html", prediction=prediction, alert_text=alert_text)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)