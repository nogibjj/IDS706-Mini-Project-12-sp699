import numpy as np
from main import data_loading, label_encoding, model_training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def test_data_loading():
    penguins = data_loading("/workspaces/IDS706-Mini-Project-12-sp699/penguins.csv")
    assert penguins is not None, "Data should not contain null values after preprocessing"

def test_label_encoding():
    penguins = data_loading("/workspaces/IDS706-Mini-Project-12-sp699/penguins.csv")
    penguins_encoded = label_encoding(penguins)
    assert isinstance(penguins_encoded['island'][0], (int, np.integer)), "Island column should be encoded as integers"
    assert isinstance(penguins_encoded['sex'][0], (int, np.integer)), "Sex column should be encoded as integers"

def test_model_training():
    penguins = data_loading("/workspaces/IDS706-Mini-Project-12-sp699/penguins.csv")

    # Apply label encoding to categorical features
    label_encoder = LabelEncoder()
    penguins['island'] = label_encoder.fit_transform(penguins['island'])
    penguins['sex'] = label_encoder.fit_transform(penguins['sex'])
    penguins['species'] = label_encoder.fit_transform(penguins['species'])

    X = penguins.drop('species', axis=1)
    y = penguins['species']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train.dropna(inplace=True)
    y_train = y_train[X_train.index]

    # Train the model
    model = model_training(X_train, y_train)

    # Test to ensure the model's random state is set correctly
    assert model.random_state == 42, "Model random state should be 42"

if __name__ == '__main__':
    test_data_loading()
    test_label_encoding()
    test_model_training()
    print("All tests passed!")