
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    
# load model
def model_fn(model_dir):
    model = DigitClassifier(input_size=784, hidden_size=10, num_classes=10)
    model.load_state_dict(torch.load(Path(model_dir) / "001" / "model.pt"))
    model.to(device).eval()
    
    return model

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    data = np.array(data).reshape(1, -1)
    print("Data shape: ", data.shape)
    return data


# inference
def predict_fn(input_object, model):
    input_object = torch.from_numpy(input_object).float() # tensors
    with torch.no_grad():
        prediction = model(input_object).detach().cpu().numpy()
        prediction = np.argmax(predictions, axis=-1)
    
    print("Prediction:", prediction)
    outputs = {"Predictions":prediction}
    
    return prediction


# postprocess
def output_fn(outputs, content_type):
    assert content_type == "application/json"
    # Print top categories per image
    return outputs
