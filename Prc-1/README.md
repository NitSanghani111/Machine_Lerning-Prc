# Practice 1 - Naive Bayes Classifier on Iris Dataset

This project implements a Naive Bayes classifier using the scikit-learn library on a manually defined subset of the Iris dataset.

---

## 🧠 What is Naive Bayes?

Naive Bayes methods are supervised learning algorithms based on applying **Bayes' theorem** with a strong (naive) assumption of **conditional independence** between features. It is particularly suited for classification tasks and works well with high-dimensional data.

---

## 📊 Dataset

A small, hardcoded sample of the Iris dataset consisting of 20 entries with the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Target class (0 or 1)

---

## 🔧 Technologies Used

- Python 3.x
- Pandas
- scikit-learn

---

## 🔍 Implementation Steps

1. **Load Data**: A custom subset of the Iris dataset is loaded as a dictionary and converted to a DataFrame.
2. **Split Data**: `train_test_split()` is used to split the dataset into training and testing sets (80/20).
3. **Train Model**: A `GaussianNB` model is created and trained on the training data.
4. **Make Predictions**: The model predicts the target for the test set.
5. **Evaluate Model**: Accuracy score, confusion matrix, and classification report are printed for evaluation.

---

## 📈 Results

The classifier performance is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report
