import os
from typing import cast

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data: about 27,000 real style customer messages with a category label
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
df = cast(pd.DataFrame, ds.to_pandas())[["instruction", "category"]]
print(df.head())
print(df["category"].value_counts())

# 2. Split: 80% to learn from, 20% to test on unseen messages
X_train, X_test, y_train, y_test = train_test_split(
    df["instruction"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"]
)

# 3. Model: TF-IDF turns text into numbers, Logistic Regression picks the category
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

# 4. Evaluate on the test set
print(classification_report(y_test, model.predict(X_test)))

# 5. Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ticket_model.joblib")
print("Saved to model/ticket_model.joblib")

