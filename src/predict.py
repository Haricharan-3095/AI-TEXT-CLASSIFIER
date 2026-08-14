import joblib


# Load trained pipeline
pipeline = joblib.load("classifier.joblib")


# Get classifier from pipeline
classifier = pipeline.named_steps["classifier"]

# Get TF-IDF vectorizer
vectorizer = pipeline.named_steps["tfidf"]


print("AI Customer Support Classifier")
print("==============================")

while True:

    text = input("\nEnter a customer message (or type 'exit'): ")

    if text.lower() == "exit":
        print("Goodbye!")
        break

    # Remove unnecessary spaces
    text = text.strip()

    if not text:
        print("Please enter a message.")
        continue

    # Convert text to TF-IDF
    X = vectorizer.transform([text])

    # Get SVM decision scores
    scores = classifier.decision_function(X)[0]

    # Find class with highest score
    best_index = scores.argmax()

    best_score = scores[best_index]

    predicted_class = classifier.classes_[best_index]

    # Unknown detection
    if best_score < -0.20:
        predicted_class = "Unknown"

    print(f"Predicted category: {predicted_class}")