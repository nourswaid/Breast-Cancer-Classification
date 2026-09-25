# Breast Cancer Classification with Logistic Regression

## Overview

This project uses the **Breast Cancer Wisconsin dataset** to build a binary classification model that predicts whether a tumor is **benign** or **malignant**.

The project follows a basic machine learning workflow:

**Data Loading → Data Cleaning → Train/Test Split → Logistic Regression → Evaluation → Cross-Validation**

## Dataset

The dataset contains measurements describing characteristics of breast cell nuclei.

The target variable is:

* **2** → Benign
* **4** → Malignant

The features used for classification are:

* Clump Thickness
* Uniformity of Cell Size
* Uniformity of Cell Shape
* Marginal Adhesion
* Single Epithelial Cell Size
* Bare Nuclei
* Bland Chromatin
* Normal Nucleoli
* Mitoses

The `Sample_code_number` column was removed because it serves as an identifier rather than a predictive feature.

## Data Preprocessing

The dataset required some cleaning before training the model.

### Missing Values

The `Bare_nuclei` column contains missing values represented by `?`.

These values were:

1. Converted to `NaN` using `pd.to_numeric()`.
2. Rows containing missing values were removed.

### Feature and Target Separation

The cleaned dataset was separated into:

* **X:** the nine tumor-related features
* **y:** the `Class` target variable

The data was then split into:

* **80% training data**
* **20% testing data**

A fixed `random_state=0` was used to make the split reproducible.

## Model

The classification model used is **Logistic Regression** from Scikit-learn.

```python
LogisticRegression(random_state=0)
```

The model was trained using the training set and then used to predict the classes of the unseen test set.

## Evaluation

The model was evaluated using two main approaches.

### Test Accuracy

The model achieved a test accuracy of:

**95.62%**

This represents the proportion of samples in the test set that were classified correctly.

### Confusion Matrix

A confusion matrix was used to examine the model's predictions for the benign and malignant classes and identify classification errors.

### 5-Fold Cross-Validation

5-fold cross-validation was performed on the training data to evaluate how consistently the model performs across different training and validation splits.

The mean cross-validation accuracy was:

**96.89%**

## Results

| Evaluation              |   Accuracy |
| ----------------------- | ---------: |
| Test Set                | **95.62%** |
| 5-Fold Cross-Validation | **96.89%** |

The cross-validation accuracy is slightly higher than the held-out test accuracy. This indicates that the model performed consistently across the validation folds, while its performance on the completely unseen test set was slightly lower.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Logistic Regression

## Project Structure

```text
breast-cancer-classification/
│
├── breast_cancer_classification.ipynb
├── breast-cancer-wisconsin.data
└── README.md
```

## What I Practiced

This project provided practice with:

* Loading and inspecting a dataset
* Handling non-standard missing values
* Converting columns to numeric values
* Removing irrelevant identifier columns
* Separating features and target variables
* Splitting data into training and testing sets
* Training a Logistic Regression classifier
* Making predictions
* Evaluating classification accuracy
* Creating a confusion matrix
* Applying 5-fold cross-validation

## Conclusion

This project demonstrates a complete classification workflow using Logistic Regression. The model achieved **95.62% accuracy on the test set** and a **96.89% mean accuracy across 5-fold cross-validation**.

The project focuses on the practical steps involved in preparing a dataset, training a classification model, and evaluating its performance using both a held-out test set and cross-validation.
