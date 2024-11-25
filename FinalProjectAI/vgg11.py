import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Define Custom VGG Model
class CustomVGG(nn.Module):
    def __init__(self, num_classes=10, extra_layers=0):
        super(CustomVGG, self).__init__()
        base_layers = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        # Add extra layers if specified
        for _ in range(extra_layers):
            base_layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
            base_layers.append(nn.BatchNorm2d(512))
            base_layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*base_layers)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CustomVGGWithKernels(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3):
        super(CustomVGGWithKernels, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Train and evaluate VGG model
# Train and evaluate VGG model
def train_and_evaluate_vgg(trainloader, testloader, device, write_to_file, name="Base VGG11", extra_layers=0):
    print(f"\nTraining and evaluating {name}...")

    # Define the model path
    model_filename = f"{name.lower().replace(' ', '_')}_model.pth"
    model_path = os.path.join("data/models", model_filename)

    # Initialize the model
    model = CustomVGG(extra_layers=extra_layers).to(device)

    # Check if the model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))  # Load the model onto the current device
    else:
        print(f"Training a new model for {name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training Loop
        model.train()
        epochs = 20
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"{name} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

        # Save the trained model
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), model_path)  # Save state_dict only
        print(f"Saved trained model as '{model_path}'")

    # Evaluation for training and testing datasets
    def evaluate(loader):
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_labels, all_preds

    # Evaluate training data
    train_labels, train_preds = evaluate(trainloader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro')

    # Evaluate testing data
    test_labels, test_preds = evaluate(testloader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')

    # Generate and write the summary table for both training and testing
    generate_vgg_summary_table(train_acc, train_precision, train_recall, train_f1, test_acc, test_precision, test_recall, test_f1, write_to_file)

    # Generate and save the confusion matrix for testing data
    target_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.title(f"{name} Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix as '{filename}'")





# Train and evaluate VGG model with adjustable kernels
def train_and_evaluate_vgg_with_kernels(
    trainloader, testloader, device, write_to_file, name="Custom VGG", kernel_size=3
):
    print(f"\nTraining and evaluating {name} with kernel size {kernel_size}...")

    # Define the model path
    model_filename = f"{name.lower().replace(' ', '_')}_kernel_{kernel_size}x{kernel_size}_model.pth"
    model_path = os.path.join("data/models", model_filename)

    # Initialize the model
    model = CustomVGGWithKernels(kernel_size=kernel_size).to(device)

    # Check if the model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from '{model_path}'...")
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location,weights_only=True))
    else:
        print(f"Training a new model for {name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training Loop
        model.train()
        epochs = 20
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"{name} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

        # Save the trained model
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved trained model as '{model_path}'")

    # Evaluation for training and testing datasets
    def evaluate(loader):
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_labels, all_preds

    # Evaluate training data
    train_labels, train_preds = evaluate(trainloader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro')

    # Evaluate testing data
    test_labels, test_preds = evaluate(testloader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')

    # Generate and write the summary table for both training and testing
    generate_vgg_summary_table(train_acc, train_precision, train_recall, train_f1, test_acc, test_precision, test_recall, test_f1, write_to_file)

    # Generate and save the confusion matrix for testing data
    target_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    filename = f"{name.lower().replace(' ', '_')}_kernel_{kernel_size}x{kernel_size}_confusion_matrix.png"
    plt.title(f"{name} Confusion Matrix (Kernel Size: {kernel_size}x{kernel_size})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix as '{filename}'")


def generate_vgg_summary_table(train_acc, train_precision, train_recall, train_f1,
                                test_acc, test_precision, test_recall, test_f1, write_to_file):
    """
    Generate and write the summary table to the results file for both training and testing data.
    """
    table_header = (
        f"\n=== VGG11 Summary Table ===\n"
        f"{'Metric':<15}{'Train':<10}{'Test':<10}\n"
        f"{'-' * 35}\n"
    )
    table_body = (
        f"{'Accuracy':<15}{train_acc:<10.4f}{test_acc:<10.4f}\n"
        f"{'Precision':<15}{train_precision:<10.4f}{test_precision:<10.4f}\n"
        f"{'Recall':<15}{train_recall:<10.4f}{test_recall:<10.4f}\n"
        f"{'F1-Score':<15}{train_f1:<10.4f}{test_f1:<10.4f}\n"
    )
    table = table_header + table_body
    print(table)
    write_to_file(table)




