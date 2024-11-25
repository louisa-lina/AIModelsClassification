# Importing required libraries and modules
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from gnb import GaussianNaiveBayes
from decision_tree import DecisionTree
from mlp import BaseMLP, DeeperMLP, ShallowerMLP, LargerHiddenMLP, SmallerHiddenMLP
from vgg11 import train_and_evaluate_vgg, train_and_evaluate_vgg_with_kernels


# Paths for saved features and models
DATA_DIR = "./data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "cifar10_features.npz")
RESULTS_FILE = "results_summary.txt"
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Utility function to write results to file
def write_to_file(text, mode='a'):
    with open(RESULTS_FILE, mode) as file:
        file.write(text + "\n")

# Utility functions for saving and loading models
def save_model(model, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# Evaluate model
def evaluate_model(y_true, y_pred, model_name, dataset_type="Test"):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    result = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    summary = (f"\n{model_name} ({dataset_type} Data) Evaluation:\n"
               f"Accuracy: {acc:.4f}\n"
               f"Precision: {precision:.4f}\n"
               f"Recall: {recall:.4f}\n"
               f"F1-Score: {f1:.4f}\n")
    print(summary)
    write_to_file(summary)
    return result
# Train and evaluate Naive Bayes
# Train and evaluate Naive Bayes with Confusion Matrices
# Train and evaluate Naive Bayes
def train_and_evaluate_naive_bayes(train_features, train_labels, test_features, test_labels):
    print("\nTraining and evaluating Naive Bayes classifiers...")
    write_to_file("\n=== Naive Bayes Results ===", mode='a')

    # Custom Gaussian Naive Bayes
    custom_model_filename = "custom_naive_bayes.pkl"
    custom_gnb = load_model(custom_model_filename)
    if custom_gnb is None:
        print("Training Custom Gaussian Naive Bayes...")
        custom_gnb = GaussianNaiveBayes()
        custom_gnb.fit(train_features, train_labels)
        save_model(custom_gnb, custom_model_filename)
    else:
        print("Loaded Custom Gaussian Naive Bayes from file.")

    # Predictions and Metrics for Training
    custom_train_preds = custom_gnb.predict(train_features)
    train_accuracy = accuracy_score(train_labels, custom_train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, custom_train_preds, average='macro'
    )

    # Predictions and Metrics for Testing
    custom_test_preds = custom_gnb.predict(test_features)
    test_accuracy = accuracy_score(test_labels, custom_test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, custom_test_preds, average='macro'
    )

    # Summary Table for Custom Naive Bayes
    table_header = (
        f"\n=== Custom Gaussian Naive Bayes Results ===\n"
        f"{'Metric':<15}{'Train':<10}{'Test':<10}\n"
        f"{'-'*40}\n"
    )
    table_body = (
        f"{'Accuracy':<15}{train_accuracy:<10.4f}{test_accuracy:<10.4f}\n"
        f"{'Precision':<15}{train_precision:<10.4f}{test_precision:<10.4f}\n"
        f"{'Recall':<15}{train_recall:<10.4f}{test_recall:<10.4f}\n"
        f"{'F1-Score':<15}{train_f1:<10.4f}{test_f1:<10.4f}\n"
    )
    table = table_header + table_body
    print(table)
    write_to_file(table)

    # Scikit-Learn Gaussian Naive Bayes
    sklearn_model_filename = "sklearn_naive_bayes.pkl"
    sklearn_gnb = load_model(sklearn_model_filename)
    if sklearn_gnb is None:
        print("Training Scikit-Learn Gaussian Naive Bayes...")
        sklearn_gnb = GaussianNB()
        sklearn_gnb.fit(train_features, train_labels)
        save_model(sklearn_gnb, sklearn_model_filename)
    else:
        print("Loaded Scikit-Learn Gaussian Naive Bayes from file.")

    # Predictions and Metrics for Training
    sklearn_train_preds = sklearn_gnb.predict(train_features)
    train_accuracy = accuracy_score(train_labels, sklearn_train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, sklearn_train_preds, average='macro'
    )

    # Predictions and Metrics for Testing
    sklearn_test_preds = sklearn_gnb.predict(test_features)
    test_accuracy = accuracy_score(test_labels, sklearn_test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, sklearn_test_preds, average='macro'
    )

    # Summary Table for Scikit-Learn Naive Bayes
    table_header = (
        f"\n=== Scikit-Learn Naive Bayes Results ===\n"
        f"{'Metric':<15}{'Train':<10}{'Test':<10}\n"
        f"{'-'*40}\n"
    )
    table_body = (
        f"{'Accuracy':<15}{train_accuracy:<10.4f}{test_accuracy:<10.4f}\n"
        f"{'Precision':<15}{train_precision:<10.4f}{test_precision:<10.4f}\n"
        f"{'Recall':<15}{train_recall:<10.4f}{test_recall:<10.4f}\n"
        f"{'F1-Score':<15}{train_f1:<10.4f}{test_f1:<10.4f}\n"
    )
    table = table_header + table_body
    print(table)
    write_to_file(table)

# Train and evaluate Decision Tree
# Train and evaluate Decision Tree
def train_and_evaluate_decision_tree(train_features, train_labels, test_features, test_labels):
    print("\nTraining and evaluating Decision Tree classifiers...")
    write_to_file("\n=== Decision Tree Results ===", mode='a')

    # Depths to test for Decision Tree
    depths = [10, 20, 30, 40, 50]

    for depth in depths:
        print(f"\nRunning Decision Tree with max_depth={depth}...")

        # Custom Decision Tree
        custom_model_filename = f"custom_decision_tree_depth_{depth}.pkl"
        custom_dt = load_model(custom_model_filename)
        if custom_dt is None:
            print("Training Custom Decision Tree...")
            custom_dt = DecisionTree(max_depth=depth)
            custom_dt.fit(train_features, train_labels)
            save_model(custom_dt, custom_model_filename)
        else:
            print(f"Loaded Custom Decision Tree from file (Depth: {depth}).")

        # Predictions and Metrics for Training
        custom_train_preds = custom_dt.predict(train_features)
        train_accuracy = accuracy_score(train_labels, custom_train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, custom_train_preds, average='macro'
        )

        # Predictions and Metrics for Testing
        custom_test_preds = custom_dt.predict(test_features)
        test_accuracy = accuracy_score(test_labels, custom_test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, custom_test_preds, average='macro'
        )

        # Generate and Save Confusion Matrix for Testing
        cm = confusion_matrix(test_labels, custom_test_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                    yticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        plt.title(f"Custom Decision Tree Confusion Matrix (Depth: {depth})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"custom_decision_tree_confusion_matrix_depth_{depth}.png")
        plt.close()

        # Summary Table for Custom Decision Tree
        table_header = (
            f"\n=== Custom Decision Tree Results (Depth: {depth}) ===\n"
            f"{'Metric':<15}{'Train':<10}{'Test':<10}\n"
            f"{'-'*40}\n"
        )
        table_body = (
            f"{'Accuracy':<15}{train_accuracy:<10.4f}{test_accuracy:<10.4f}\n"
            f"{'Precision':<15}{train_precision:<10.4f}{test_precision:<10.4f}\n"
            f"{'Recall':<15}{train_recall:<10.4f}{test_recall:<10.4f}\n"
            f"{'F1-Score':<15}{train_f1:<10.4f}{test_f1:<10.4f}\n"
        )
        table = table_header + table_body
        print(table)
        write_to_file(table)

        # Scikit-Learn Decision Tree
        sklearn_model_filename = f"sklearn_decision_tree_depth_{depth}.pkl"
        sklearn_dt = load_model(sklearn_model_filename)
        if sklearn_dt is None:
            print("Training Scikit-Learn Decision Tree...")
            sklearn_dt = DecisionTreeClassifier(max_depth=depth)
            sklearn_dt.fit(train_features, train_labels)
            save_model(sklearn_dt, sklearn_model_filename)
        else:
            print(f"Loaded Scikit-Learn Decision Tree from file (Depth: {depth}).")

        # Predictions and Metrics for Training
        sklearn_train_preds = sklearn_dt.predict(train_features)
        train_accuracy = accuracy_score(train_labels, sklearn_train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, sklearn_train_preds, average='macro'
        )

        # Predictions and Metrics for Testing
        sklearn_test_preds = sklearn_dt.predict(test_features)
        test_accuracy = accuracy_score(test_labels, sklearn_test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, sklearn_test_preds, average='macro'
        )

        # Generate and Save Confusion Matrix for Testing
        cm = confusion_matrix(test_labels, sklearn_test_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                    yticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        plt.title(f"Scikit-Learn Decision Tree Confusion Matrix (Depth: {depth})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"sklearn_decision_tree_confusion_matrix_depth_{depth}.png")
        plt.close()

        # Summary Table for Scikit-Learn Decision Tree
        table_header = (
            f"\n=== Scikit-Learn Decision Tree Results (Depth: {depth}) ===\n"
            f"{'Metric':<15}{'Train':<10}{'Test':<10}\n"
            f"{'-'*40}\n"
        )
        table_body = (
            f"{'Accuracy':<15}{train_accuracy:<10.4f}{test_accuracy:<10.4f}\n"
            f"{'Precision':<15}{train_precision:<10.4f}{test_precision:<10.4f}\n"
            f"{'Recall':<15}{train_recall:<10.4f}{test_recall:<10.4f}\n"
            f"{'F1-Score':<15}{train_f1:<10.4f}{test_f1:<10.4f}\n"
        )
        table = table_header + table_body
        print(table)
        write_to_file(table)

# Train and evaluate MLP
def train_and_evaluate_mlp(train_features, train_labels, test_features, test_labels, model_class):
    print(f"\nTraining and evaluating {model_class.__name__}...")
    write_to_file(f"\n=== {model_class.__name__} Results ===", mode='a')

    # Convert features and labels to tensors and move them to the device
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Load or initialize the model
    model_filename = f"{model_class.__name__}_mlp_model.pkl"
    model = load_model(model_filename)
    if model is None:
        model = model_class().to(device)  # Move model to device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training Loop
        model.train()
        epochs = 20
        epoch_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        save_model(model, model_filename)
        print(f"Saved {model_class.__name__} model to {model_filename}")

        # Save learning curve
        plt.figure()
        plt.plot(range(epochs), epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_class.__name__} Learning Curve")
        plt.legend()
        plt.savefig(f"{model_class.__name__}_learning_curve.png")
        plt.close()
    else:
        print(f"Loaded {model_class.__name__} model from {model_filename}")
        model.to(device)  # Ensure loaded model is moved to the correct device

    # Evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_features)
        test_outputs = model(test_features)

        _, train_predictions = torch.max(train_outputs, 1)
        _, test_predictions = torch.max(test_outputs, 1)

        # Metrics Calculation for Training
        train_accuracy = accuracy_score(train_labels.cpu(), train_predictions.cpu())
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels.cpu(), train_predictions.cpu(), average='macro'
        )

        # Metrics Calculation for Testing
        test_accuracy = accuracy_score(test_labels.cpu(), test_predictions.cpu())
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels.cpu(), test_predictions.cpu(), average='macro'
        )

        # Summary Table
        table_header = (
            f"\n=== {model_class.__name__} Summary Table ===\n"
            f"{'Metric':<15}{'Train':<15}{'Test':<15}\n"
            f"{'-'*45}\n"
        )
        table_body = (
            f"{'Accuracy':<15}{train_accuracy:<15.4f}{test_accuracy:<15.4f}\n"
            f"{'Precision':<15}{train_precision:<15.4f}{test_precision:<15.4f}\n"
            f"{'Recall':<15}{train_recall:<15.4f}{test_recall:<15.4f}\n"
            f"{'F1-Score':<15}{train_f1:<15.4f}{test_f1:<15.4f}\n"
        )
        table = table_header + table_body
        print(table)
        write_to_file(table)

        # Confusion Matrix for Test Data
        cm = confusion_matrix(test_labels.cpu(), test_predictions.cpu())
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                    yticklabels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        plt.title(f"{model_class.__name__} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"{model_class.__name__}_confusion_matrix.png")
        plt.close()


# Dataset setup
def setup_cifar10():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    def filter_dataset(dataset, num_per_class):
        class_count = {i: 0 for i in range(10)}
        filtered_indices = []
        for idx, (_, label) in enumerate(dataset):
            if class_count[label] < num_per_class:
                filtered_indices.append(idx)
                class_count[label] += 1
            if all(count >= num_per_class for count in class_count.values()):
                break
        return torch.utils.data.Subset(dataset, filtered_indices)

    trainset_filtered = filter_dataset(trainset, 500)
    testset_filtered = filter_dataset(testset, 100)

    trainloader = torch.utils.data.DataLoader(trainset_filtered, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset_filtered, batch_size=64, shuffle=False)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()

    return trainloader, testloader, model

# Feature extraction and PCA
def extract_and_pca(trainloader, testloader, model):
    if os.path.exists(PROCESSED_DATA_PATH):
        data = np.load(PROCESSED_DATA_PATH)
        return data["train_features"], data["train_labels"], data["test_features"], data["test_labels"]

    def extract_features(dataloader, model):
        features, labels = [], []
        with torch.no_grad():
            for inputs, lbls in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = outputs.view(outputs.size(0), -1)
                features.append(outputs.cpu())
                labels.append(lbls)
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = extract_features(trainloader, model)
    test_features, test_labels = extract_features(testloader, model)

    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features.numpy())
    test_features_pca = pca.transform(test_features.numpy())

    np.savez(PROCESSED_DATA_PATH,
             train_features=train_features_pca,
             train_labels=train_labels.numpy(),
             test_features=test_features_pca,
             test_labels=test_labels.numpy())

    return train_features_pca, train_labels.numpy(), test_features_pca, test_labels.numpy()

# Add these functions near the top of `main.py`
def inspect_pkl_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    print("Loaded `.pkl` model:")
    print(model)

def inspect_pth_model(file_path):
    model_state_dict = torch.load(file_path)
    print("Loaded `.pth` model's state dictionary:")
    for key, value in model_state_dict.items():
        print(f"{key}: {value.shape}")


# Main function
def main():
    write_to_file("=== Results Summary ===\n", mode='w')

    trainloader, testloader, model = setup_cifar10()
    train_features_pca, train_labels, test_features_pca, test_labels = extract_and_pca(trainloader, testloader, model)

    while True:
        print("\nChoose which part to run:")
        print("1. Naive Bayes")
        print("2. Decision Tree")
        print("3. Base MLP")
        print("4. Deeper MLP")
        print("5. Shallower MLP")
        print("6. Larger Hidden MLP")
        print("7. Smaller Hidden MLP")
        print("8. Base VGG11")
        print("9. VGG11 + 2 Layers")
        print("10. VGG11 + 4 Layers")
        print("11. VGG11 with 3x3 Kernels")
        print("12. VGG11 with 5x5 Kernels")
        print("0. Exit")
        choice = input("Enter your choice (1/2/3/4/5/6/7/8/9/10/11/12/0): ")

        if choice == "1":
            train_and_evaluate_naive_bayes(train_features_pca, train_labels, test_features_pca, test_labels)
        elif choice == "2":
            train_and_evaluate_decision_tree(train_features_pca, train_labels, test_features_pca, test_labels)
        elif choice == "3":
            train_and_evaluate_mlp(train_features_pca, train_labels, test_features_pca, test_labels, BaseMLP)
        elif choice == "4":
            train_and_evaluate_mlp(train_features_pca, train_labels, test_features_pca, test_labels, DeeperMLP)
        elif choice == "5":
            train_and_evaluate_mlp(train_features_pca, train_labels, test_features_pca, test_labels, ShallowerMLP)
        elif choice == "6":
            train_and_evaluate_mlp(train_features_pca, train_labels, test_features_pca, test_labels, LargerHiddenMLP)
        elif choice == "7":
            train_and_evaluate_mlp(train_features_pca, train_labels, test_features_pca, test_labels, SmallerHiddenMLP)
        elif choice == "8":
            train_and_evaluate_vgg(trainloader, testloader, device, write_to_file, name="Base VGG11", extra_layers=0)
        elif choice == "9":
            train_and_evaluate_vgg(trainloader, testloader, device, write_to_file, name="VGG11 + 2 Layers", extra_layers=2)
        elif choice == "10":
            train_and_evaluate_vgg(trainloader, testloader, device, write_to_file, name="VGG11 + 4 Layers", extra_layers=4)
        elif choice == "11":
            train_and_evaluate_vgg_with_kernels(trainloader, testloader, device, write_to_file, name="VGG11 with 3x3 Kernels", kernel_size=3)
        elif choice == "12":
            train_and_evaluate_vgg_with_kernels(trainloader, testloader, device, write_to_file, name="VGG11 with 5x5 Kernels", kernel_size=5)
        elif choice == "0":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")





if __name__ == "__main__":
    main()

