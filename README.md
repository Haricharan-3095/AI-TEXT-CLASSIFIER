🤖 AI Customer Support Classifier

An end-to-end Machine Learning application that automatically classifies customer support messages into different categories.

The project covers the complete ML deployment workflow:

- Data preprocessing
- Exploratory data analysis
- Model training
- Model evaluation
- Hyperparameter tuning
- Confidence threshold tuning
- FastAPI REST API
- Web UI
- Docker containerization
- Docker Hub
- Cloud deployment with Render

🚀 Live Demo

https://ai-text-classifier.onrender.com

🎯 Project Overview

Customer support teams receive a large number of messages every day. Manually categorizing these messages can be time-consuming.

This application automatically analyzes a customer message and predicts its category.

Supported Categories

- Account Access
- Billing
- General Inquiry
- Technical Issue

🧠 Machine Learning Model

The application uses:

**TF-IDF Vectorization**

to convert customer messages into numerical features.

A:

**Logistic Regression**

classifier is then used to predict the category.

The trained model and vectorizer are saved using Joblib.

🔄 Machine Learning Pipeline

```text
Customer Support Dataset
          ↓
Data Preprocessing
          ↓
TF-IDF Vectorization
          ↓
Logistic Regression
          ↓
Model Evaluation
          ↓
Hyperparameter Tuning
          ↓
Final Model
          ↓
FastAPI
          ↓
Web UI


🌐 Application Architecture
                 Customer
                    │
                    ▼
                Web UI
                    │
                    ▼
              FastAPI API
                    │
                    ▼
             TF-IDF Vectorizer
                    │
                    ▼
          Logistic Regression
                    │
                    ▼
            Predicted Category



📡 API
Prediction Endpoint
POST /predict

Example request:

{
    "text": "I forgot my password"
}

Example response:

{
    "category": "Account Access"
}
Health Check
GET /

The API returns a response confirming that the service is running.

🖥️ Web Interface

The project includes a web interface where users can:
Enter a customer support message.
Submit the message.
Send it to the FastAPI backend.
Receive the predicted category.

🐳 Docker

The application is containerized using Docker.
Build the image:
docker build -t ai-text-classifier .
Run the container:
docker run -p 8000:8000 ai-text-classifier
Then open:
http://localhost:8000

☁️ Cloud Deployment
The application is deployed using:
Docker
Docker Hub
Render

Live application:
https://ai-text-classifier.onrender.com

🛠️ Technologies Used

Technology	Purpose
Python	Programming language
Pandas	Data processing
Scikit-learn	Machine Learning
TF-IDF	Text feature extraction
Logistic Regression	Classification
Joblib	Model serialization
FastAPI	REST API
Uvicorn	API server
HTML/CSS/JavaScript	Web UI
Docker	Containerization
Docker Hub	Container registry
Render	Cloud deployment


📁 Project Structure
ai-text-classifier/
│
├── data/
│   └── customer_support.csv
│
├── src/
│   ├── api.py
│   ├── data_exploration.py
│   ├── train.py
│   ├── tune.py
│   ├── tune_threshold.py
│   ├── final_train.py
│   ├── predict.py
│   └── test_confidence.py
│
├── static/
│   └── index.html
│
├── tests/
│   └── test_api.py
│
├── classifier.joblib
├── vectorizer.joblib
├── model.joblib
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md


🔮 Future Improvements
Increase the size of the training dataset.
Add more customer support categories.
Improve handling of unknown messages.
Add prediction confidence scores.
Add authentication.
Add database integration.
Add monitoring and logging.
Experiment with transformer-based NLP models.
Implement CI/CD.

👨‍💻 Author
Hari Charan Uggirala
```
