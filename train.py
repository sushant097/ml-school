
import os
import argparse

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import accuracy_score


import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Expects a .csv file where the first column is the label
# and returns a tensor dataset, the input_size and the number of classes
def csv_to_tensor(file):
    X_train = pd.read_csv(Path(file), header=None)
    y_train = X_train.iloc[:,0].values
    X_train = X_train.iloc[:, 1:].values
    tensor_labels = torch.from_numpy(y_train).long()
    tensor_data = torch.from_numpy(X_train).float()
    return TensorDataset(tensor_data, tensor_labels)

# DataLoader used for mini-batch training
# DataLoader used for mini-batch training
def make_data_loader(train_file, val_file, batch_size):
    train_dataset = csv_to_tensor(train_file)
    val_dataset = csv_to_tensor(val_file)
    
    input_size = len(val_dataset.tensors[0][0])
    print(" input_size ",input_size)
    tensor_labels = val_dataset.tensors[1]
    print(" tensor_labels ", tensor_labels)
    labels = set(label.item() for label in tensor_labels)
    num_classes = len(labels)
    print("num_classes ", num_classes)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True)
    return train_loader, val_loader, input_size, num_classes

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
    

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return accuracy_score(labels.cpu(), classes.cpu())

def train_model(model, train_dataset, validation_dataset, loss_fn, optimizer, batch_size=100, n_epochs=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    training_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.00
        running_accuracy = 0.00
        for i, (x_batch, y_batch) in enumerate(train_dataset):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # zero gradients
            optimizer.zero_grad()
            # predictions
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
            running_loss += loss.item()
            running_accuracy += accuracy(outputs, y_batch)
        
        running_loss /= len(train_dataset)
        running_accuracy /= len(train_dataset)

        # validation accuracy
        model.eval()
        correct, total = 0.0, 0.0
        for images, labels in validation_dataset:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum() 
            
        val_accuracy =  correct/total
        print(f'Epoch: {epoch+1}/{n_epochs} - '+
                  f'loss: {loss:.4f} - '+ 
                  f'accuracy: {running_accuracy:.4f} - '+
                  f'val_accuracy: {val_accuracy:.4f}')
            
    
    return model, training_losses

            
# def predict(model, input_tensor):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     input_tensor.to(device)
    
#     with torch.no_grad():
#         raw_output = model(input_tensor)
#         output_probabilities = model.probabilities(raw_output)
#         pred_category = torch.argmax(output_probabilities).item()
#         pred_probability = torch.max(output_probabilities).item()
#     return pred_category, round(pred_probability, 4)

# def evaluate(model, test_dataset):
#     actual = []
#     predicted = []
#     n_correct = 0
#     incorrect = []
    
#     for i, (x, y) in enumerate(test_dataset):
#         actual.append(y.item())
#         label, prob = predict(model, x)
#         predicted.append(label)
        
#         if (label == y.item()):
#             n_correct+=1
#         else:
#             incorrect.append([i, y.item(), label])
    
#     print(f'Accuracy on test set: {n_correct/len(test_dataset)*100:.2f}%')

#     confusion_matrix = metrics.confusion_matrix(actual, predicted)
#     print(f"Confusion Matrix: {confusion_matrix}")
    

# Hyperparameters
hidden_size = 10
learning_rate = 0.001


def train(base_directory, train_path, validation_path, epochs=50, batch_size=32):
    train_loader, val_loader, input_size, num_classes = make_data_loader(Path(train_path) / "train.csv", 
                                                                      Path(validation_path) / "validation.csv", 
                                                                      batch_size)
   
    model = DigitClassifier(input_size, hidden_size, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, training_loss = train_model(model, 
                                    train_loader,
                                    val_loader,
                                    loss_fn, 
                                    optimizer,
                                    batch_size,
                                    epochs)
        
    model_filepath = Path(base_directory) / "model" / "001" 
    
    model_filepath.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_filepath / "model.pt")
    
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))
    # model.eval()
    
if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to the entry point
    # as script arguments. SageMaker will also provide a list of special parameters
    # that you can capture here. Here is the full list: 
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default="/opt/ml/")
    parser.add_argument("--train_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation_path", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    args, _ = parser.parse_known_args()
    
    train(
        base_directory=args.base_directory,
        train_path=args.train_path,
        validation_path=args.validation_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
