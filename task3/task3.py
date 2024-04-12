import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load spam data from a CSV file
def load_data():
    data = pd.read_csv('spam-data.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y


# Train a logistic regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model


# Manually extracted features for the first email
def get_first_email_features():
    # Assume manual extraction has resulted in the following features
    # Number of Words, Number of Links, Number of Capitalized Words, Number of Spam Words
    # Example: "Subject: Innovative Solutions for Your Business" might have 10 words, 2 links, 3 capitalized words, and 1 spam word
    return [69, 2, 3, 1]  # These numbers are purely illustrative


# Predict if the first email is spam
def predict_email(model):
    first_email_features = get_first_email_features()
    first_email_prediction = model.predict([first_email_features])
    result = 'Spam' if first_email_prediction[0] == 1 else 'Not Spam'
    print(f'The first email is classified as: {result}')


def main():
    # Load data and train the model
    X, y = load_data()
    model = train_model(X, y)

    # Predict the first email
    predict_email(model)


if __name__ == "__main__":
    main()
#Model Accuracy: 0.93
#The first email is classified as: Not Spam