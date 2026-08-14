import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/customer_support.csv")

X = df["text"]
y = df["label"]


# ============================================================
# SAME DATA SPLIT USED DURING TRAINING
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


# ============================================================
# LOAD TRAINED PIPELINE
# ============================================================

pipeline = joblib.load("classifier.joblib")

classifier = pipeline.named_steps["classifier"]
vectorizer = pipeline.named_steps["tfidf"]


# ============================================================
# GET VALIDATION SCORES
# ============================================================

X_val_tfidf = vectorizer.transform(X_val)

scores = classifier.decision_function(X_val_tfidf)

predictions = classifier.classes_[scores.argmax(axis=1)]

best_scores = scores.max(axis=1)


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    -0.20,
    -0.10,
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


print("\nThreshold Tuning")
print("================")

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Unknown':<12}"
)


for threshold in thresholds:

    final_predictions = []

    for prediction, score in zip(
        predictions,
        best_scores
    ):

        if score < threshold:
            final_predictions.append("Unknown")
        else:
            final_predictions.append(prediction)

    accuracy = accuracy_score(
        y_val,
        final_predictions
    )

    unknown_count = final_predictions.count("Unknown")

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.2%}"
        f"{unknown_count:<12}"
    )