import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Setup & hyperparameters
batch_size = 256
epochs = 50     # Change this for different number of tests
learning_rate = 0.01
dropoutr = 0.5  # Dropout rate

# Load data (Fashion-MNIST)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Download the full datasets
full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Grab only the first 5000 images to speed up training and force overfitting
subset_indices = list(range(5000))
train_subset = torch.utils.data.Subset(full_train_dataset, subset_indices)

# Load the new subset instead of the full dataset
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Definition
class SimpleMLP(nn.Module):
    def __init__(self, use_dropout=False, dropout_rate=dropoutr):
        super(SimpleMLP, self).__init__()
        self.use_dropout = use_dropout
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        if self.use_dropout: x = self.drop1(x)
            
        x = self.fc2(x)
        x = self.relu2(x)
        if self.use_dropout: x = self.drop2(x)
            
        x = self.fc3(x)
        return x

# Training and Evaluation Function
def train_and_evaluate(model, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_acc_history = []
    test_acc_history = []
    
    print(f"--- Training {name} ---")
    for epoch in range(epochs):
        # Training
        model.train() # Sets model to training mode
        correct_train, total_train = 0, 0
        
        for images, labels in train_loader:
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        train_acc_history.append(train_acc)
        
        # Evaluation
        model.eval() # Sets model to evaluation mode
        correct_test, total_test = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = 100 * correct_test / total_test
        test_acc_history.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return train_acc_history, test_acc_history

# Run the experiments
baseline_model = SimpleMLP(use_dropout=False)
drop_model = SimpleMLP(use_dropout=True, dropout_rate=dropoutr)

base_train_acc, base_test_acc = train_and_evaluate(baseline_model, "Baseline (No Dropout)")
drop_train_acc, drop_test_acc = train_and_evaluate(drop_model, f"Model (With Dropout {dropoutr})")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(base_train_acc, label='Baseline Train Acc', linestyle='--', color='red')
plt.plot(base_test_acc, label='Baseline Test Acc', color='red')
plt.plot(drop_train_acc, label='Dropout Train Acc', linestyle='--', color='blue')
plt.plot(drop_test_acc, label='Dropout Test Acc', color='blue')

plt.title('Baseline vs Dropout')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Highlights the gap 
plt.fill_between(range(len(base_train_acc)), base_train_acc, base_test_acc, color='red', alpha=0.1)
plt.fill_between(range(len(drop_train_acc)), drop_train_acc, drop_test_acc, color='blue', alpha=0.1)

plt.show()

