import pandas as pd
import re

positive_words = set([
    "good", "great", "love", "excellent", "amazing", "wonderful", "best",
    "fantastic", "enjoy", "enjoyed", "beautiful", "brilliant", "perfect",
    "happy", "favorite", "awesome", "superb", "delightful", "impressive",
    "recommend", "entertaining", "funny", "charming", "masterpiece"
])

negative_words = set([
    "bad", "terrible", "hate", "awful", "worst", "boring", "waste",
    "poor", "disappointing", "horrible", "stupid", "dull", "annoying",
    "mess", "fails", "failure", "lame", "weak", "predictable", "cliche",
    "mediocre", "disappointed", "ridiculous", "pointless"
])

def predict(text):
    words = re.findall(r"\b\w+\b", text.lower())
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    return 1 if pos_count >= neg_count else 0

def evaluate(csv_path):
    df = pd.read_csv(csv_path)
    preds = df['text'].apply(predict)
    accuracy = (preds == df['label']).mean()
    return accuracy, df, preds

if __name__ == "__main__":
    accuracy, df, preds = evaluate('data/test.csv')
    print(f"Rule-based accuracy: {accuracy:.4f}")

    wrong = df[preds != df['label']].head(5)
    print("\nSample misclassified reviews:")
    for i, row in wrong.iterrows():
        print(f"\nTrue label: {row['label']}, Predicted: {preds[i]}")
        print(row['text'][:200], "...")
