import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# TODO: add necessary import

from ml.model import train_model, inference

def get_test_numbers():
    #Get numbers for testing
    X = np.array([[1,2,3],[10,15,20], [11,13,17],[4,8,16]])
    y = np.array([0,1,0,1])
    return X,y

def get_sample_data():
    data = pd.DataFrame({
        'age': [55,50,45,40],
        'education' : ['Masters', 'Bachelors', 'Doctorate', 'Masters'],
        'workclass' : ['Government', 'Private', 'Government', 'Government'],
        'salary' : ['<=50K', '<=50K', '>50K', '<=50K']
    })
    return data



# TODO: implement the first test. Change the function name and input as needed
def test_model_return_type():
    """
    # Test to verify RandomForest model works
    """
    X,y = get_test_numbers()
    model = train_model(X,y)

    assert isinstance(model, RandomForestClassifier), \
        "should return RandomForestClassifier instance"


# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    #Test the inference function
    """
    X,y = get_test_numbers()
    model = train_model(X,y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray), \
      "inference should return numpy array"

# TODO: implement the third test. Change the function name and input as needed
def test_split():
    """
    #Test the data split size
    """
    train, test = train_test_split(get_sample_data(), test_size = 0.20, random_state=42)

    total_size = len(get_sample_data())
    train_split = len(train)
    test_split = len(test)

    assert train_split + test_split == total_size, \
      "the split should equal the size of the data"