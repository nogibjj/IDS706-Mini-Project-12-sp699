![MLflow](https://github.com/nogibjj/IDS706-Mini-Project-12-sp699/actions/workflows/cicd.yml/badge.svg)
# IDS-706-Data-Engineering :computer:

## Mini Project 12 :page_facing_up: 

## :ballot_box_with_check: Requirements
* Create a simple machine-learning model.</br>
* Use MLflow to manage the project, including tracking metrics.</br>

## :ballot_box_with_check: To-do List
* __Data processing functionality__: Create a simple machine learning model and evaluate its functionality.</br>
* __MLflow tracking__: Record the model implemented through MLflow tracking.

## :ballot_box_with_check: Dataset
`penguins.csv`
  <img src="https://github.com/nogibjj/Suim-Park-Mini-Project-2/assets/143478016/fe1c7646-539f-4bd5-ba5f-c67f47cbc4c9.png" width="600" height="400"/>
  - Data were collected and made available by __Dr. Kristen Gorman__ and the __Palmer Station__, Antarctica LTER, a member of the Long Term Ecological Research Network. It shows three different species of penguins observed in the Palmer Archipelago, Antarctica.
  - [penguins.csv](https://github.com/nogibjj/IDS706-Mini-Project-12-sp699/raw/main/penguins.csv)
* `Description of variables`</br>
  <img src="https://github.com/nogibjj/Suim-Park-Mini-Project-2/assets/143478016/6b0020de-5499-43ea-b6d6-a67f52aa8d58.png" width="350" height="450"/></br>
  - In this dataset, we use the characteristics of penguins to infer their species.

## :ballot_box_with_check: Main Progress
#### `Section 1` Data Load
##### Data is being loaded to adjust the machine learning model.
* `main.py`
```Python
def data_loading(filepath):
    return pd.read_csv(filepath)
```

#### `Section 2` Label Encoding
##### The 'island' and 'sex' columns of the given data are processed using LabelEncoder for label encoding.
* `main.py`
```Python
def label_encoding(data):
    label_encoder = LabelEncoder()
    data['island'] = label_encoder.fit_transform(data['island'])
    data['sex'] = label_encoder.fit_transform(data['sex'])
    return data
```

#### `Section 3` Model Testing
##### The model is trained to execute the machine learning algorithm.
* `main.py`
```Python
def model_training(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
```

#### `Section 4` Main
##### Run the machine learning model and test it.
* `main.py`
```Python
def main():
    # Data loading
    penguins = data_loading("penguins.csv")
    
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
```

