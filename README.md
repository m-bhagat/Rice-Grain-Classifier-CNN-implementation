# Rice-Grain-Classifier-CNN-implementation

# CNN Model Documentation for Rice Classification

## Overview
This code implements a Convolutional Neural Network (CNN) to classify images of rice into five categories: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. The code is optimized to run efficiently on laptops without a powerful GPU.

## Key Features
- Custom implementation of the VGG16-inspired architecture.
- Dataset split into training, validation, and testing sets.
- Optimized for smaller computational resources with reduced image sizes and simplified layers.
- Performance evaluation through confusion matrix, classification report, and training-validation metrics visualization.

## Steps to Run

### 1. Set up Conda Environment
Create a new Conda environment and install the necessary dependencies.

```bash
conda create -n rice_classifier python=3.8
conda activate rice_classifier
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib scikit-learn seaborn
conda install jupyter notebook spyder
```

### 2. Dataset Preparation
Organize the dataset into folders named after the classes:
```
data/
    Arborio/
    Basmati/
    Ipsala/
    Jasmine/
    Karacadag/
```
Ensure the images are placed in their respective folders.

### 3. Code Highlights

#### Data Splitting
The dataset is split into training (60%), validation (20%), and testing (20%) subsets. Each class's images are moved to appropriate directories.

#### CNN Architecture
The CNN architecture includes:
- Three convolutional layers with ReLU activation and max-pooling.
- A fully connected layer with dropout for regularization.
- Output layer for class prediction.

#### Training
- Optimized using Adam optimizer.
- Categorical cross-entropy loss for multi-class classification.

#### Evaluation
- Accuracy, precision, recall, and F1-score metrics are calculated.
- Confusion matrix and loss/accuracy plots provide insights into model performance.

### 4. Visualization
#### Sample Images
A 5x5 grid showcases 5 sample images from each category before training begins.

#### Confusion Matrix
The confusion matrix displays the performance of the classifier for each category.

#### Accuracy and Loss Graphs
Two graphs are plotted:
1. Training vs Validation Loss.
2. Validation Accuracy across epochs.

### 5. Results
#### Sample Classification Report:
```
              precision    recall  f1-score   support

     Arborio       0.96      0.94      0.95       200
     Basmati       0.99      0.98      0.99       200
      Ipsala       1.00      0.99      0.99       200
     Jasmine       0.97      0.99      0.98       200
   Karacadag       0.96      0.97      0.97       200

    accuracy                           0.98      1000
   macro avg       0.98      0.98      0.98      1000
weighted avg       0.98      0.98      0.98      1000
```

#### Test Accuracy:
Achieved a high test accuracy of 98%.

### 6. How to Save and Load Model
Save the trained model:
```python
torch.save(model.state_dict(), 'rice_classifier_cnn.pth')
```
Load the model for inference:
```python
model.load_state_dict(torch.load('rice_classifier_cnn.pth'))
model.eval()
```

### 7. Optimizations
- Reduced image size to 128x128.
- Simplified CNN layers to reduce computational requirements.
- Small batch size and fewer epochs for efficient training.

### 8. Conclusion
The code is a robust solution for rice classification that balances accuracy and resource efficiency, making it suitable for execution on devices with limited hardware capabilities.

### 9. Citations

Rice Image Dataset
DATASET: https://www.muratkoklu.com/datasets/

Citation Request: See the articles for more detailed information on the data.

Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

Cinar, I., & Koklu, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. Selcuk Journal of Agriculture and Food Sciences, 35(3), 229-243. https://doi.org/10.15316/SJAFS.2021.252

Cinar, I., & Koklu, M. (2022). Identification of Rice Varieties Using Machine Learning Algorithms. Journal of Agricultural Sciences https://doi.org/10.15832/ankutbd.862482

Cinar, I., & Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering, 7(3), 188-194. https://doi.org/10.18201/ijisae.2019355381

---
Feel free to adapt the code further based on your dataset or hardware setup!



