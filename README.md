# Fruit Classification Using Transfer Learning

This notebook trains a deep learning model for fruit classification using transfer learning from pretrained models (VGG16, ResNet50, MobileNet).

## 📂 Dataset

The dataset used is **Fruits-360** with 141 different fruit categories. It is stored in Google Drive and extracted for training.

## 🛠️ Requirements

Before running the notebook, install the necessary dependencies:

```sh
pip install tensorflow keras matplotlib numpy seaborn scikit-learn
```

## 🚀 How to Use

1. **Mount Google Drive**: The dataset is loaded from Google Drive.
2. **Preprocess the Dataset**:
   - Extract the dataset.
   - Split into training, validation, and test sets.
3. **Feature Extraction**:
   - Use VGG16, ResNet50, and MobileNet to extract features.
4. **Train a Classifier**:
   - Use a dense classifier on top of extracted features.
   - Implement early stopping and checkpointing.
5. **Evaluate Performance**:
   - Generate accuracy scores and confusion matrices.

## 🏆 Models Used

- **VGG16**
- **ResNet50**
- **MobileNet**

## 📊 Metrics

- Training and validation accuracy
- Confusion matrix
- Classification report

## 🎨 Visualization

To better understand the model's decision-making, **Grad-CAM** is used to highlight important regions in the images that contribute to the model's predictions.

## 🔗 Additional Notes

- The dataset is automatically split into 80% training and 20% validation.
- Hyperparameters (learning rate, dropout, patience) are optimized per model.
- The best model is saved as `first_model_<MODEL_NAME>.keras`.

  # Fruit Classification Using Transfer Learning

This notebook trains a deep learning model for fruit classification using transfer learning from pretrained models (VGG16, ResNet50, MobileNet).

## 📂 Dataset

The dataset used is **Fruits-360** with 141 different fruit categories. It is stored in Google Drive and extracted for training.

## 🛠️ Requirements

Before running the notebook, install the necessary dependencies:

```sh
pip install tensorflow keras matplotlib numpy seaborn scikit-learn
```

## 🚀 How to Use

1. **Mount Google Drive**: The dataset is loaded from Google Drive.
2. **Preprocess the Dataset**:
   - Extract the dataset.
   - Split into training, validation, and test sets.
3. **Feature Extraction**:
   - Use VGG16, ResNet50, and MobileNet to extract features.
4. **Train a Classifier**:
   - Use a dense classifier on top of extracted features.
   - Implement early stopping and checkpointing.
5. **Evaluate Performance**:
   - Generate accuracy scores and confusion matrices.

## 🏆 Models Used

- **VGG16**
- **ResNet50**
- **MobileNet**

## 📊 Metrics

- Training and validation accuracy
- Confusion matrix
- Classification report

## 🎨 Visualization

To better understand the model's decision-making, **Grad-CAM** is used to highlight important regions in the images that contribute to the model's predictions.
![image](https://github.com/user-attachments/assets/2ad6d052-67b8-4fe6-be8d-6ddc4c7db612)
![image](https://github.com/user-attachments/assets/85f7a55b-7fd4-4f92-8575-3dabaabf342f)

## 🔗 Additional Notes

- The dataset is automatically split into 80% training and 20% validation.
- Hyperparameters (learning rate, dropout, patience) are optimized per model.
- The best model is saved as `first_model_<MODEL_NAME>.keras`.




