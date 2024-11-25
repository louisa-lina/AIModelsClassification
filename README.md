# AI Models Classification with CIFAR-10

This project involves the analysis of CIFAR-10 image dataset using multiple machine learning and deep learning models. The main goal is to compare the performance of various models (Naive Bayes, Decision Tree, MLP variants, and CNN) in terms of accuracy, precision, recall, F1-score, and confusion matrices.

**IMPORTANT**: Due to the large size of the models for the CNN they have been stored in Google Drive and can be accessed [here](https://drive.google.com/drive/folders/1hVRO0UWX01fnux2EFVApngWGwpbueJ0t?usp=sharing)

---

## How to Run the Project

1. **Environment Setup**:
   - Ensure you have Python 3.x installed.
   - Install the required libraries by running:
     ```bash
     pip install torch torchvision scikit-learn matplotlib seaborn
     !pip install scikit-learn torch
     !pip install torch torchvision
     !pip install scikit-learn
     !pip install numpy
     ```
     

2. **Google Colab**:
   - Upload the project files to Google Drive.
   - Create a Notebook and Open the `.ipynb` file in Google Colab.
   - Ensure GPU runtime is enabled:
     - Navigate to `Runtime > Change runtime type`.
     - Select `GPU` under the "Hardware accelerator" section.
    -Import Project zip file
    - Ensure to unzip the file
    - cd to project directory
      
3. **Run the Code**:
   - Execute
       ```bash
     !python main.py
     ```
   

4. **Results**:
   - Outputs such as metrics, confusion matrices will be saved in the `results_summary.txt` file and `.png` image files for the confusion matrices.
   - Model are stored in the models subfolder that can be found in the data folder.

---

## Project Highlights

1. **Models Evaluated**:
   - Naive Bayes (Custom and Scikit-Learn variants)
   - Decision Trees with varying depths
   - Multilayer Perceptron (MLP) with 5 architecture variances.
   - Convolutional Neural Network (CNN) based on VGG11.
     
2. **Features**:
   - Feature extraction was performed using a pre-trained ResNet18 model.
   - Dimensionality reduction was implemented using Principal Component Analysis (PCA).

3. **Metrics**:
   - Accuracy, precision, recall, F1-score, and confusion matrices were calculated for both training and testing datasets.

4. **Optimizations**:
   - Stochastic Gradient Descent (SGD) optimizer with momentum for MLPs and CNNs.
  

5. **Best Results**:
   - **MLP with Larger Hidden Layers** achieved the best testing accuracy of 79.9%.

---

## Author

- **Louisa-Lina Meziane (40133119)**

---

## Academic Context

- **Course**: COMP 472 - Artificial Intelligence
- **Instructor**: Dr. Kefaya Qaddoum
- **Institution**: Concordia University
- **Semester**: Fall 2024



