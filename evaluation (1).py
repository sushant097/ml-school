
import os
import json
import tarfile
import numpy as np
import pandas as pd

from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


MODEL_PATH = "/opt/ml/processing/model/"
TEST_PATH = "/opt/ml/processing/test/"
OUTPUT_PATH = "/opt/ml/processing/evaluation/"


# Our model, the heart
class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size) 
        self.activation_1 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size) 
        self.activation_2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.probabilities = nn.Softmax(dim=0)
    
    def forward(self, x):
        hidden_1 = self.hidden_layer_1(x)
        hidden_activated_1 = self.activation_1(hidden_1)
        hidden_2 = self.hidden_layer_2(hidden_activated_1)
        hidden_activated_2 = self.activation_2(hidden_2)
        out_layer = self.output_layer(hidden_activated_2)
        return out_layer
    
def evaluate(model_path, test_path, output_path):
    # The first step is to extract the model package provided
    # by SageMaker.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))
        
    # We can now load the model from disk.
    # model = keras.models.load_model(Path(model_path) / "001")
    model = DigitClassifier(input_size=784, hidden_size=10, num_classes=10)
    model.load_state_dict(torch.load(Path(model_path) / "001" / "model.pt"))
    model.eval()
    
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test.iloc[:, 0].values
    X_test = X_test.iloc[:, 1:].values
    
    predictions = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy}")

    # Let's add the accuracy of the model to our evaluation report.
    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
        },
    }
    
    # We need to save the evaluation report to the output path.
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path=MODEL_PATH, 
        test_path=TEST_PATH,
        output_path=OUTPUT_PATH
    )
