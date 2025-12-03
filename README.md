# MNIST Digit Classification – README

This project builds a complete machine learning workflow for classifying handwritten digits from the **MNIST dataset** using multiple models and ensemble learning techniques.

---

## 📌 **Project Overview**

This notebook performs:

* MNIST data fetching
* Visualization of sample digits
* Class distribution analysis
* Feature scaling
* Training/test splitting
* Evaluation of multiple classification algorithms
* Cross-validation using StratifiedKFold
* Ensemble learning via Voting Classifier
* Model comparison using accuracy boxplots

---

## 📁 **Dataset: MNIST**

The MNIST dataset consists of **70,000 grayscale images** of handwritten digits (0–9), each sized **28×28 pixels**.

Each sample has:

* **784 pixel values** (flattened 28×28 image)
* **1 label** (digit 0–9)

Loaded using:

```python
mnist = fetch_openml('mnist_784', version=1)
```

---

## 🧭 **Workflow Summary**

### 1. Import Required Packages

Includes NumPy, Pandas, Seaborn, Matplotlib, Scikit-Learn, and XGBoost.

### 2. Load and Inspect the Dataset

* Check structure and keys
* Display image samples
* Show class distribution (balanced dataset)

### 3. Visualizations

* Display single digit image
* Grid of 30 sample images using a custom `print_image()` function
* Countplot for digit class frequencies

### 4. Split the Dataset

To speed up training:

* Take 50% of MNIST → `X_small`, `y_small`
* Train/test split on reduced dataset

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

---

## ⚙️ **Preprocessing**

### 🔹 Feature Scaling

Standardization using:

```python
StandardScaler()
```

This helps gradient-based and distance-based classifiers.

---

## 🧠 **Models Used**

This project evaluates multiple classifiers:

### **Individual Models**

* Logistic Regression
* Gaussian Naive Bayes
* Random Forest Classifier
* Gradient Boosting Classifier
* K-Nearest Neighbors (KNN)

### **Ensemble Model**

A **Voting Classifier** combining:

* Logistic Regression
* Random Forest
* Gradient Boosting

Uses **soft voting** for improved performance.

---

## 📝 **Model Evaluation**

### 🔹 Cross-Validation

Performed using **StratifiedKFold (5 splits)** to maintain class balance.
Scores are calculated using **accuracy** metric.

### 🔹 Test Set Evaluation

Each model prints:

* Classification report
* Precision, recall, F1-score
* Overall accuracy

### 🔹 Comparison Plot

A seaborn boxplot displays CV accuracy distribution across all models.

---

## 📊 **Visual Outputs**

The notebook generates:

* Image visualizations (single + multiple)
* Class distribution plot
* Cross-validation accuracy comparison boxplot

---

## 🎯 **Goal of the Project**

To evaluate and compare traditional machine learning approaches for MNIST classification **without using deep learning**, demonstrating:

* Strong baseline model performance
* Benefits of ensemble learning
* Practical ML workflow on image datasets

---

## 🧩 **Technologies Used**

* Python 3
* NumPy & Pandas
* Matplotlib & Seaborn
* Scikit-Learn
* XGBoost (imported but not used)

---

## 🚀 **How to Run the Notebook**

1. Install required libraries:

   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn xgboost
   ```
2. Run the notebook cell-by-cell.
3. Ensure internet access is available (required to fetch MNIST from OpenML).

---

## ✨ **Author**

This project demonstrates classic machine learning techniques applied to MNIST digit recognition.

---

If you want, I can also generate:

* A combined README for both projects
* A GitHub-ready version with badges & images
* A PDF version of the documentation
