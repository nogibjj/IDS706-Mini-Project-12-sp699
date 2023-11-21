from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import mlflow
import mlflow.sklearn

def data_loading(filepath):
    return pd.read_csv(filepath)

def label_encoding(data):
    label_encoder = LabelEncoder()
    data['island'] = label_encoder.fit_transform(data['island'])
    data['sex'] = label_encoder.fit_transform(data['sex'])
    return data

def model_training(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Data loading
    penguins = data_loading("/workspaces/IDS706-Mini-Project-12-sp699/penguins.csv")
    
    # Label encoding
    penguins = label_encoding(penguins)

    X = penguins.drop('species', axis=1)
    y = penguins['species']

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train.dropna(inplace=True)
    y_train = y_train[X_train.index]
    X_test.dropna(inplace=True)
    y_test = y_test[X_test.index]

    # Model train
    model = model_training(X_train, y_train)

    # MLflow track
    with mlflow.start_run():
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()