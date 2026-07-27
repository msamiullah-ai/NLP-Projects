import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"TF-IDF + Logistic Regression accuracy: {accuracy:.4f}")

# Save misclassified examples for the report
test_df = test_df.reset_index(drop=True)
wrong_mask = preds != y_test.values
wrong = test_df[wrong_mask].head(5)
print("\nSample misclassified reviews:")
for i, row in wrong.iterrows():
    print(f"\nTrue label: {row['label']}, Predicted: {preds[i]}")
    print(row['text'][:200], "...")
