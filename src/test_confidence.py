import joblib


# Load trained model
pipeline = joblib.load("classifier.joblib")


# Test messages
messages = [
    "I forgot my password",
    "My credit card was charged twice",
    "The application keeps crashing",
    "How can I contact support?",
    "hi",
    "hello",
    "good morning",
    "asdfgh",
    "what's the weather?"
]


# Get the classifier inside the pipeline
classifier = pipeline.named_steps["classifier"]


# Convert text into TF-IDF vectors
vectorizer = pipeline.named_steps["tfidf"]

X = vectorizer.transform(messages)


# Get SVM decision scores
scores = classifier.decision_function(X)


# Get predictions
predictions = pipeline.predict(messages)


# Display results
print("\nConfidence Analysis")
print("===================")

for message, prediction, score in zip(
    messages,
    predictions,
    scores
):

    print("\nText:", message)
    print("Predicted:", prediction)
    print("Scores:")

    for label, value in zip(
        classifier.classes_,
        score
    ):
        print(f"  {label}: {value:.4f}")