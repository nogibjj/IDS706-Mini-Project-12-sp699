from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import mlflow
import mlflow.sklearn

def main():
    # Load the dataset
    penguins = pd.read_csv("/workspaces/IDS706-Mini-Project-12-sp699/penguins.csv")
    
    label_encoder = LabelEncoder()
    penguins['island'] = label_encoder.fit_transform(penguins['island'])
    penguins['sex'] = label_encoder.fit_transform(penguins['sex'])

    print(penguins.dtypes)

    # Define your features and target variable
    X = penguins.drop('species', axis=1)
    y = penguins['species']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train.dropna(inplace=True)
    y_train = y_train[X_train.index]

    X_test.dropna(inplace=True)
    y_test = y_test[X_test.index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run():

        # Log parameters (here we can log hyperparameters of the model)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()