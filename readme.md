# Project Overview

It is divided into two parts, the pytorch part and the c++ part. The convolutional neural network (CNN) is implemented through the PyTorch framework and C++, and the image classification is performed on the CIFAR-10 dataset using the ResNet architecture. It includes data enhancement technology (Cutout) and the python implementation part in Jupyter Notebook.

The c++ part uses C++ to implement the core functions of the convolution operation and the fully connected layer.

## File structure

```
project-root/
│
├── checkpoint/ # Folder for saving model checkpoints
├── dataset/ # File containing the CIFAR-10 dataset
├── utils/ # Utility scripts, including data loading and augmentation
├── final.cpp # Convolution and fully connected layer operations implemented in C++
├── train.ipynb # Jupyter Notebook for training models
├── test (1)(1).ipynb # Jupyter Notebook for testing models
├── readme.md # Project description document
```

---

## Detailed code description

### 1. final.cpp

**Function**:
The core operations in the neural network are implemented in C++, including 2D convolution, ReLU activation, fully connected layers, and loading weights for inference. This file shows how to implement the basic functions of convolutional neural networks from scratch without relying on deep learning libraries.

**Key functions**:

1. `conv2d`: performs two-dimensional convolution operations, supporting multi-channel input and output.

2. `relu`: ReLU activation function, used to introduce nonlinear characteristics, setting all negative values ​​to 0.

3. `fully_connected`: implementation of the fully connected layer, flattens the feature map after convolution, and inputs it into the fully connected layer for linear transformation.

4. `load_weights`: loads pre-trained model weights from a file for subsequent inference.



### 2. ResNet.py

**Function**:
Implements the definition of ResNet architecture and supports ResNet with different layers. ResNet is a deep residual network that introduces "skip connections" to alleviate the gradient vanishing problem in deep networks.

**Key modules**:

1. `BasicBlock`: Basic residual module for ResNet18 and ResNet34.

2. `Bottleneck`: Deeper modules for ResNet50 and above.

3. `_make_layer`: Build multiple residual blocks and adjust the network depth.



### 3. train.ipynb

**Function**:
The train.ipynb file implements the training process of the ResNet18 model, loads the CIFAR-10 dataset, performs data augmentation operations, and uses PyTorch's SGD optimizer for model training
**Key functions**:
Use the read_dataset function to load training, validation, and test data from the CIFAR-10 dataset and apply Cutout data augmentation.

Load the ResNet18 model and modify its last fully connected layer to adapt to the 10-category classification task of CIFAR-10.

Train the model on the training set, calculate the loss value, and evaluate it on the validation set, using CrossEntropyLoss as the loss function and dynamically adjusting the learning rate.

Save the model weights when the validation loss decreases.

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the dataset
batch_size = 128
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')

# Define the model
model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Train model
for epoch in range(1, n_epochs + 1):
model.train()
for data, target in train_loader:
data, target = data.to(device), target.to(device)
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

---

### 3. test (1)(1).ipynb

**Function**:
test (1)(1).ipynb file is responsible for loading the trained model, evaluating it on the test set, calculating the accuracy of the model, and saving the quantized 8-bit model.

**Code snippet and explanation**:

# Load the trained model
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model.eval()

# Quantize the model
model_int8 = torch.quantization.quantize_dynamic(
model.cpu(),
{nn.Linear}, # Quantize the fully connected layer
dtype=torch.qint8 # Quantize to 8 bits
)

# Evaluate the model on the test set
model = model.to(device)
total_sample = 0
right_sample = 0
for data, target in test_loader:
data, target = data.to(device), target.to(device)
output = model(data)
_, pred = torch.max(output, 1)
correct_tensor = pred.eq(target.data.view_as(pred))
total_sample += data.size(0)
right_sample += correct_tensor.sum().item() print(f"Accuracy: {100 * right_sample / total_sample}%") # Save the quantized model torch.save(model_int8.state_dict(), 'checkpoint/resnet18_cifar10_int8.pt') ``` - This class increases the randomness of training data and reduces overfitting by randomly generating patches (removed areas) on each image.

# How to run

1. **Environment setup**:
- Download the CIFAR-10 dataset to the `dataset/` folder, install the pytorch environment and the ipykernel kernel of jupyter.

2. **Train the model**:
- Open and run `train.ipynb` to complete the data loading, model definition and training steps.

3. **Test the model**:
- Use the `test (1)(1).ipynb` notebook to load the trained model and evaluate its performance on the test set.
The C++ part compiles an exe executable file
---