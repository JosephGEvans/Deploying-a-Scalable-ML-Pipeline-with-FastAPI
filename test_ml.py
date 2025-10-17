import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from train_model import cat_features, label, X_train, X_test, y_train, y_test, model


@pytest.fixture(scope="session")
def raw_data():
    """
    Create a Pandas DataFrame object with the expected raw data input.
    """
    df = pd.read_csv("./data/census.csv")
    return df


def test_raw_data_features(raw_data):
    """
    Ensure the raw input data has the expected feature and label columns used in processing and training.
    """
    # Create a set of actual feature names
    raw_data_features = set(raw_data.columns)
    # Create a set of expected feature names
    expected_features = set(cat_features)
    # Add the label to the set of expected feature names
    expected_features.add(label)

    # Test that all the expected features exist in the raw data
    assert expected_features.issubset(raw_data_features)


def test_processed_data_consistency():
    """
    Verifies the training and test data sets have consistent feature width and data types.
    """
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.dtype == X_test.dtype
    assert y_train.dtype == y_test.dtype


def test_model_algorithm():
    """
    The model algorithm is expected to be a Random Forest Classifier
    """
    assert isinstance(model, RandomForestClassifier)
