import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


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
# 3. SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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


print("\nDataset Split")
print("-------------")
print(f"Training examples:   {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Test examples:       {len(X_test)}")


# ============================================================
# 4. FINAL MODEL
# ============================================================
#
# Selected during hyperparameter tuning:
#
# Algorithm: Linear SVM
# C: 0.5
# ngram_range: (1, 1)
#
# ============================================================

final_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 1)
        )
    ),
    (
        "classifier",
        LinearSVC(
            C=0.5
        )
    )
])


# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model...")

final_pipeline.fit(
    X_train,
    y_train
)

print("Final model training completed!")


# ============================================================
# 6. FINAL TEST PREDICTIONS
# ============================================================

y_test_pred = final_pipeline.predict(
    X_test
)


# ============================================================
# 7. TEST PREDICTION ANALYSIS
# ============================================================

print("\nFinal Test Prediction Analysis")
print("-------------------------------")

for text, actual, predicted in zip(
    X_test,
    y_test,
    y_test_pred
):

    status = "✓" if actual == predicted else "✗"

    print(f"{status} Text: {text}")
    print(f"  Actual:    {actual}")
    print(f"  Predicted: {predicted}")
    print()


# ============================================================
# 8. FINAL TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\nFinal Test Evaluation")
print("---------------------")
print(f"Test Accuracy: {test_accuracy:.2%}")


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=labels
)

print("\nConfusion Matrix")
print("----------------")

print("Labels:")
print(labels)

print("\nMatrix:")
print(cm)


# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_pipeline,
    "classifier.joblib"
)

print("\nFinal model saved as classifier.joblib")