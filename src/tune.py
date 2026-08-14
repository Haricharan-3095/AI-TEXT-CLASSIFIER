import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


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
# 3. TRAIN / VALIDATION / TEST SPLIT
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


print("\nDataset Split")
print("-------------")
print(f"Training examples:   {len(X_train)}")
print(f"Validation examples: {len(X_val)}")
print(f"Test examples:       {len(X_test)}")


# ============================================================
# 4. HYPERPARAMETERS
# ============================================================

C_values = [0.1, 0.5, 1, 2, 5, 10]

ngram_ranges = [
    (1, 1),
    (1, 2),
    (1, 3)
]


# ============================================================
# 5. STORE RESULTS
# ============================================================

results = []

best_accuracy = 0
best_C = None
best_ngram = None


# ============================================================
# 6. EXPERIMENT
# ============================================================

print("\nHyperparameter Tuning")
print("=====================")

for ngram_range in ngram_ranges:

    for C in C_values:

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range
                )
            ),
            (
                "classifier",
                LinearSVC(
                    C=C
                )
            )
        ])

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Validation prediction
        y_val_pred = pipeline.predict(
            X_val
        )

        # Validation accuracy
        accuracy = accuracy_score(
            y_val,
            y_val_pred
        )

        results.append(
            {
                "C": C,
                "ngram_range": ngram_range,
                "accuracy": accuracy
            }
        )

        print(
            f"C={C:<4} "
            f"n-gram={ngram_range} "
            f"Accuracy={accuracy:.2%}"
        )

        # Track best configuration
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_C = C
            best_ngram = ngram_range


# ============================================================
# 7. BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)

print(f"C:              {best_C}")
print(f"n-gram range:   {best_ngram}")
print(f"Validation Acc: {best_accuracy:.2%}")