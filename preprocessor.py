
import os
import numpy as np
import pandas as pd
import tempfile

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIR = "/opt/ml/processing"
DATA_FILEPATH_TRAIN = Path(BASE_DIR) / "input" / "mnist_train.csv"
DATA_FILEPATH_TEST = Path(BASE_DIR) / "input" / "mnist_test.csv"


def save_splits(base_dir, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """
    
    train_path = Path(base_dir) / "train" 
    validation_path = Path(base_dir) / "validation" 
    test_path = Path(base_dir) / "test"
    
    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)
    
    
def preprocess(base_dir, train_data_filepath, test_data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train, validation,
    and a test set.
    """
    
    train_df = pd.read_csv(train_data_filepath)
    test_df = pd.read_csv(test_data_filepath)
    
    X_train = train_df.drop(['label'], axis=1)
    X_train = X_train.values/255.
    X_test = test_df.drop(['label'], axis=1)
    X_test = X_test.values/255.
    y_train = train_df['label'].values.astype('int')
    y_test = test_df['label'].values.astype('int')
    
    # validation set
    train, validation = np.split(X_train, [int(.8 * len(X_train))])

    validation_split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=46)
    validation_split.split(X_train, y_train)
    training_idx, validation_idx = list(validation_split.split(X_train, y_train))[0]
    
    x_training = X_train[training_idx]
    y_training = y_train[training_idx]

    x_validation = X_train[validation_idx]
    y_validation = y_train[validation_idx]
    
    
    train = np.concatenate((np.expand_dims(y_training, axis=1), x_training), axis=1)
    validation = np.concatenate((np.expand_dims(y_validation, axis=1), x_validation), axis=1)
    test = np.concatenate((np.expand_dims(y_test, axis=1), X_test), axis=1)
    
    
    save_splits(base_dir, train, validation, test)
        

if __name__ == "__main__":
    preprocess(BASE_DIR, DATA_FILEPATH_TRAIN, DATA_FILEPATH_TEST)
