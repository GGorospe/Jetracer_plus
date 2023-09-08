# Tutorial code for Pytorch
# About: this code demonstrates creation and training of a machine learning model for a classification task.
#        Key points in this example include:
#                 - Use of the torchvision datasets module for downloading vision datasets
#                 - Use of the torch.utilis.data module's DataLoader to supply the data to the model during draining
#                 - Creation of a custom NeuralNetwork function for defining the network
#                 - A train function for training the model
#                 - A test function to test the accuracy of the model during the course of the training

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets # This module contains many dataset objects that are automatically downloaded if needed
from torchvision.transforms import ToTensor


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Passing the downloaded dataset to DataLoader which adds an index for iterating over the dataset.
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(training_data,batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# From this we see that theshape of X is a 28x28x1 representation of a image in tensor form with 64 bit precision
# We also see that y, the label, is a 64 bit integer corresponding to the label category index

# Now creating the model: definition of a neural network.
# This will be done by creating a custom class "NeuralNetwork"

# get cpu, gpu or mps device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# how to know we're using our GPU at maximum speed?

# Define model:
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

# As can be seen from the print statement, the model contains 5 layers, the first takes the 784 pixels from the image,
# This is followed by a Rectified Linear Unit, that leaves activated units as 1, and rounds unactivated units to 0.00.
# The ReLU helps speed up training
# The final layer takes the 512 features and produces a 10 feature output.

# Next, to train a model, we need a loss function and an optimizer.
loss_fn = nn.CrossEntropyLoss()
# The optimizer tries to find the best possible set of parameters for the model using some criterion to determin the best option.
# THe Stochastic Gradient Descent and the Adam (most common) are available.
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# start the training

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Training Done!")
