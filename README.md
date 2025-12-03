# 📘 Codveda Data Science Internship — Iris Dataset Tasks

Welcome to the repository! This project contains **three structured Google Colab notebooks**, each representing one required task for the Codveda Data Science Internship. All tasks use the **Iris dataset**, a clean and well-known dataset ideal for machine learning and deep learning.

This README provides:

* Overview of each task
* Methodology used
* Files included
* How to run each notebook
* Summary of results

---

# 🌸 Dataset

The **Iris dataset** consists of 150 samples of iris flowers with the following numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species (target)

This dataset is perfectly balanced (3 classes × 50 samples each) and contains no missing values.

---

# 📂 Project Structure

```
📁 codveda-iris-tasks
│
├── 📓 1_iris_classification_logreg.ipynb
├── 📓 2_iris_clustering_kmeans.ipynb
├── 📓 3_iris_neural_network_keras.ipynb
│
├── 📝 README.md (this file)
└── 🌸 iris.csv
```

---

# ✅ Task 1 — Logistic Regression Classification

**Notebook:** `1_iris_classification_logreg.ipynb`

### 🔍 Objective

Build a **multiclass classification model** to predict iris flower species using Logistic Regression.

### 📌 Workflow

1. Data Understanding (info, describe, missing values, label distribution)
2. Visual Exploration (pairplot, heatmap)
3. Preprocessing

   * Label encoding
   * Feature scaling (StandardScaler)
   * Train-test split
4. Model Training (Logistic Regression)
5. Evaluation

   * Accuracy, Precision, Recall, F1-score
   * Confusion Matrix
   * Classification Report

### 🟢 Result Summary

* Logistic Regression achieves **high accuracy (>90%)**.
* Petal length & petal width are the strongest predictors.

---

# ✅ Task 2 — K-Means Clustering

**Notebook:** `2_iris_clustering_kmeans.ipynb`

### 🔍 Objective

Group iris samples **without labels** using K-Means clustering.

### 📌 Workflow

1. EDA (pairplot, heatmap)
2. Preprocessing (scaling)
3. Optimal K determination

   * Elbow Method
   * Silhouette Score
4. Final Clustering (K=3)
5. PCA visualization
6. Cluster vs Species comparison

### 🟢 Result Summary

* K=3 is the optimal number of clusters.
* Clusters align well with actual species, especially for Setosa.

---

# ✅ Task 3 — Neural Network Classification (Keras)

**Notebook:** `3_iris_neural_network_keras.ipynb`

### 🔍 Objective

Build a **simple feed-forward neural network** to classify iris species.

### 📌 Workflow

1. Data Understanding & EDA
2. Preprocessing

   * Label Encoding → One-hot encoding
   * Scaling
   * Train-test split
3. Model Architecture

   * Dense(16, ReLU)
   * Dense(8, ReLU)
   * Dense(3, Softmax)
4. Training (30 epochs)
5. Evaluation

   * Accuracy & Loss curves
   * Test accuracy

### 🟢 Result Summary

* Neural Network achieves **excellent accuracy (~95%)**.
* Model trains quickly due to small dataset.

---

# ▶️ How to Run the Notebooks

You can open each notebook in **Google Colab**:

1. Upload the `.ipynb` file and `iris.csv`
2. Run each cell in order
3. Ensure TensorFlow is installed for the neural network notebook

---

# 📌 Conclusion

Using the Iris dataset, we successfully completed **three Data Science tasks** covering:

* Classical Machine Learning
* Unsupervised Learning
* Deep Learning

These notebooks demonstrate skills in:

* Data exploration
* Preprocessing
* Modeling
* Evaluation
* Visualization

They are ready for Codveda submission and suitable for GitHub portfolio use.

---

# 💬 Contact

If you need improvements or extended versions, feel free to update this repository or reach out.

Happy Learning & Coding! 🚀
