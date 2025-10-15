import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_FILE = 'my_dataset.csv'  # Ensure this matches your filename
TEXT_COL = 'problem_statement'
TAG_COL = 'problem_tags'

# List of tags you want the crawler to find relevant
TARGET_TAGS = ['dp', 'implementation', 'greedy', 'math']

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found in the current directory.")
        return

    print("--- Loading and Cleaning Data ---")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=[TEXT_COL, TAG_COL])
    
    # Logic fix: Check if ANY of the target tags are in the problem_tags string
    print(f"Targeting problems containing any of: {TARGET_TAGS}")
    df['label'] = df[TAG_COL].apply(
        lambda x: 1 if any(tag in str(x).lower() for tag in TARGET_TAGS) else 0
    )

    print(f"Found {df['label'].sum()} relevant problems out of {len(df)} total.")

    # Convert text to numerical vectors
    print("Vectorizing text content...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df[TEXT_COL])
    y = df['label']

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    print("Training the classifier (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Report accuracy
    predictions = model.predict(X_test)
    print(f"Training Complete! Test Accuracy: {accuracy_score(y_test, predictions):.2%}")

    # Save the trained brain
    joblib.dump(model, 'classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Success: 'classifier.pkl' and 'vectorizer.pkl' have been created.")

if __name__ == "__main__":
    train_model()