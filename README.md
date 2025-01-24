## Crop Disease Classification

### Overview
This project focuses on classifying crop diseases using images. The dataset contains 17,938 labeled images spanning five categories:
- **Cassava Bacterial Blight (CBB)**
- **Cassava Brown Streak Disease (CBSD)**
- **Cassava Green Mottle (CGM)**
- **Cassava Mosaic Disease (CMD)**
- **Healthy**

The aim was to develop a machine learning model capable of predicting whether a crop has a disease or is healthy based on its image.

### Dataset
- **Source**: [Kaggle Crop Diseases Classification Dataset](https://www.kaggle.com/datasets/mexwell/crop-diseases-classification/data)
- The dataset consists of images stored in the `train_images` folder and labels in a JSON file.
- **Label Distribution**:
  - **CBB**: 921
  - **CBSD**: 1,831
  - **CGM**: 1,993
  - **CMD**: 11,027
  - **Healthy**: 2,166

### Approaches and Results

#### 1. ResNet50 on a Balanced Dataset
- CMD was undersampled, and other classes were oversampled to 3,000 images each.
- **Results**: Good accuracy but poor macro average due to overfitting on CMD.

#### 2. ViT with Gaussian Blur
- Applied Gaussian blur for noise reduction.
- **Results**: Underperformed compared to ResNet50, highlighting inefficiency for this task.

#### 3. CNN with Rotatory Augmentation
- Implemented augmentation with CNN.
- **Results**: Worse than other models, showing poor adaptability.

#### 4. ResNet50 without Augmentation
- Best results achieved by undersampling CMD and using 25 epochs.
- **Accuracy**: High, but the model remained overfitted on CMD.

#### 5. ResNext50 with Undersampling
- CMD class was undersampled to 2,000 images.
- **Results**: Achieved a macro average of 0.73 but dropped in accuracy due to reduced CMD bias.

#### 6. ResNet152 with Augmentation
- Augmented the least represented class to 1,000 images.
- **Results**: Slight overfitting led to lower macro averages compared to ResNext50.

### Conclusion
- **Best Model**: ResNext50 achieved a macro average of 0.73 with minimal overfitting.
- Balancing datasets and preventing biases towards dominant classes significantly improved the results.
- Future work can involve additional data types, advanced neural architectures, and cross-validation to ensure robustness.

### Future Directions
- Explore multi-modal data integration for enhanced accuracy.
- Implement comprehensive cross-validation.
- Investigate additional architectures and transfer learning techniques.

### How to Run
1. Clone the repository.
2. Install required dependencies using `requirements.txt`.
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model on the test dataset:
   ```bash
   python evaluate.py
   ```

### Acknowledgments
Special thanks to the Kaggle community for providing the dataset and fostering research in crop disease detection.
