import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/customer_support.csv")

print("Dataset loaded successfully!")
print(f"Total examples: {len(df)}")


# ============================================================
# 2. SEPARATE INPUTS AND LABELS
# ============================================================

X = df["text"]
y = df["label"]


# ============================================================
# 3. SPLIT DATASET
# ============================================================

# 70% Training
# 15% Validation
# 15% Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# 4. DISPLAY DATASET SPLIT
# ============================================================

print("\nDataset Split")
print("-------------")
print(f"Training examples:   {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Test examples:       {len(X_test)}")


# ============================================================
# 5. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ]),

    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ]),

    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC())
    ])
}


# ============================================================
# 6. TRAIN AND EVALUATE EACH MODEL
# ============================================================

results = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Validation prediction
    y_val_pred = model.predict(X_val)

    # Accuracy
    accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    results[name] = accuracy

    print(f"\nValidation Accuracy: {accuracy:.2%}")

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_val_pred,
            zero_division=0
        )
    )

    # Prediction analysis
    print("Prediction Analysis")
    print("-------------------")

    for text, actual, predicted in zip(
        X_val,
        y_val,
        y_val_pred
    ):

        status = "✓" if actual == predicted else "✗"

        print(f"{status} {text}")
        print(f"   Actual:    {actual}")
        print(f"   Predicted: {predicted}")
        print()


# ============================================================
# 7. COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, accuracy in results.items():
    print(f"{name:<25} {accuracy:.2%}")


# ============================================================
# 8. FIND BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=results.get
)

best_accuracy = results[best_model_name]

print("\nBest Model")
print("----------")
print(f"Model: {best_model_name}")
print(f"Validation Accuracy: {best_accuracy:.2%}")