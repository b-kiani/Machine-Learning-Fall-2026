# Machine Learning

<p align="center">
  <strong>Practical Machine Learning with Python, Jupyter & Scikit-Learn</strong>
</p>

<p align="center">
  Learn the complete machine learning workflow — from raw datasets to reproducible models, evaluation, and final projects.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter\&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?logo=googlecolab\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Science-013243?logo=numpy\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas\&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-green)

</p>

---

## 📌 About This Repository

This repository is a practical, project-based introduction to **Machine Learning with Python**.

The course focuses on developing the skills required to take a machine learning problem from:

```text
Problem
   ↓
Dataset
   ↓
Data Preparation
   ↓
Exploratory Analysis
   ↓
Model Selection
   ↓
Training
   ↓
Evaluation
   ↓
Interpretation
   ↓
Reproducible Pipeline
   ↓
Final Project
```

Rather than focusing only on theory, the repository combines **short lectures, hands-on coding laboratories, notebooks, datasets, exercises, mini-projects, and a final capstone project**.

---

## 🎯 Learning Goals

By completing this course, you should be able to:

* Prepare and explore datasets using Python.
* Load, clean, transform, and split datasets.
* Identify common machine learning problem types.
* Train basic supervised and unsupervised learning models.
* Select an appropriate model for a given problem.
* Evaluate models using suitable performance metrics.
* Identify overfitting and basic model limitations.
* Build reproducible machine learning workflows.
* Communicate machine learning results through notebooks and reports.
* Present and defend a machine learning project.

---

## 🧠 Topics Covered

### 1. Machine Learning Foundations

* What is Machine Learning?
* Machine Learning applications
* Supervised vs. unsupervised learning
* Datasets and features
* Training and testing data
* Basic ML workflow
* Python for data analysis

### 2. Data Preparation

* Loading datasets
* Pandas fundamentals
* Exploring rows and columns
* Missing values
* Categorical variables
* Feature preparation
* Train/test splitting
* Data leakage
* Basic exploratory data analysis

### 3. Regression

* Regression problems
* Linear Regression
* Predictions
* Regression errors
* MAE
* MSE
* RMSE
* Interpreting regression results

### 4. Classification

* Classification problems
* Logistic Regression
* Binary classification
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### 5. k-Nearest Neighbors

* Distance-based learning
* Choosing `k`
* Feature scaling
* Training a k-NN classifier
* Evaluating predictions

### 6. Decision Trees

* Tree-based learning
* Decision rules
* Tree depth
* Overfitting
* Model interpretation

### 7. Naive Bayes

* Basic probability concepts
* Naive Bayes classification
* Simple text/data classification
* Advantages and limitations

### 8. Model Comparison

* Comparing multiple models
* Selecting appropriate metrics
* Interpreting model performance
* Evidence-based conclusions

### 9. Clustering

* Unsupervised learning
* k-Means clustering
* Selecting the number of clusters
* Visualizing clusters
* Interpreting cluster meaning

### 10. Machine Learning Pipelines

* Data preprocessing
* Model training
* Evaluation
* Reproducible workflows
* Scikit-learn pipelines
* Clean notebook design

### 11. Machine Learning Projects

* Problem definition
* Dataset selection
* Model selection
* Experimentation
* Model comparison
* Results analysis
* Technical reporting
* Presentation and viva

---

## 🗓️ Course Roadmap

|   Week | Topic                            | Practical Work                              |
| -----: | -------------------------------- | ------------------------------------------- |
| **01** | Introduction to Machine Learning | Tools, notebooks, datasets                  |
| **02** | Python for Data Work             | Pandas, CSV files, visualization            |
| **03** | Data Preparation                 | Missing values, features, train/test split  |
| **04** | Linear Regression                | MAE, MSE, RMSE                              |
| **05** | Classification                   | Logistic Regression, classification metrics |
| **06** | k-Nearest Neighbors              | Scaling and k-NN                            |
| **07** | **Midterm Mini-Project**         | Notebook + demonstration                    |
| **08** | Decision Trees                   | Depth, overfitting, interpretation          |
| **09** | Naive Bayes                      | Classification applications                 |
| **10** | Model Comparison                 | Compare multiple models                     |
| **11** | k-Means Clustering               | Clustering and visualization                |
| **12** | ML Pipelines                     | Reproducible workflows                      |
| **13** | Capstone Planning                | Dataset + problem definition                |
| **14** | Capstone Development             | Testing, feedback, presentation             |
| **15** | **Final Capstone**               | Report + code + presentation                |

---

## 🧪 Practical Learning

Every major concept is accompanied by practical implementation.

### Typical Notebook Workflow

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Load dataset
df = pd.read_csv("data/dataset.csv")

# 3. Inspect data
print(df.head())
print(df.info())

# 4. Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6. Train a model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📊 Machine Learning Models

The core models covered in this repository include:

| Model               | Type         | Main Application            |
| ------------------- | ------------ | --------------------------- |
| Linear Regression   | Supervised   | Regression                  |
| Logistic Regression | Supervised   | Classification              |
| k-NN                | Supervised   | Classification              |
| Decision Tree       | Supervised   | Classification / Regression |
| Naive Bayes         | Supervised   | Classification              |
| k-Means             | Unsupervised | Clustering                  |

---

## 📏 Model Evaluation

Choosing a model is not enough — you must understand **how well it performs**.

### Regression

Common metrics:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error

### Classification

Common metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 🔬 Projects

### 🟢 Midterm Mini-Project

The midterm project focuses on solving a basic machine learning problem using a complete workflow:

```text
Dataset
   ↓
Exploration
   ↓
Preparation
   ↓
Model Selection
   ↓
Training
   ↓
Evaluation
   ↓
Interpretation
```

**Deliverables:**

* Jupyter/Colab notebook
* Short report
* Dataset preparation
* Exploratory analysis
* Model implementation
* Evaluation
* Short demonstration

The project emphasizes problem understanding, model selection, evaluation, and limitations.

---

### 🔵 Final Capstone Project

The final project integrates the complete machine learning workflow.

Students define a problem, select a dataset, develop models, compare results, and communicate their findings.

**Deliverables:**

* Dataset
* Notebook / source code
* Reproducible ML pipeline
* Written report
* Results and visualizations
* Model comparison
* Final presentation
* Viva / technical questions

The capstone emphasizes problem understanding, model comparison, technical implementation, reproducibility, reporting, presentation, and teamwork.

---

## 🧩 Recommended Project Structure

```text
machine-learning/
│
├── README.md
│
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_python_data_work.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_linear_regression.ipynb
│   ├── 05_classification.ipynb
│   ├── 06_knn.ipynb
│   ├── 07_midterm_project.ipynb
│   ├── 08_decision_trees.ipynb
│   ├── 09_naive_bayes.ipynb
│   ├── 10_model_comparison.ipynb
│   ├── 11_kmeans.ipynb
│   └── 12_ml_pipeline.ipynb
│
├── labs/
│   ├── lab01/
│   ├── lab02/
│   ├── lab03/
│   └── ...
│
├── assignments/
│   ├── midterm/
│   └── capstone/
│
├── datasets/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── reports/
│   ├── midterm/
│   └── final/
│
└── requirements.txt
```

---

## 🛠️ Technology Stack

### Programming

* Python 3.x

### Development Environment

* Jupyter Notebook
* JupyterLab
* Google Colab

### Core Libraries

```text
NumPy
Pandas
Matplotlib
Scikit-Learn
```

### Supporting Tools

* Git
* GitHub
* Kaggle
* Google Colab

The course materials specifically identify Python/Jupyter/Colab together with NumPy, pandas, matplotlib, scikit-learn, and GitHub as the primary software environment.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/machine-learning.git
cd machine-learning
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Or open the notebooks directly in **Google Colab**.

---

## 📦 Example `requirements.txt`

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## 🧠 Knowledge Checks

Use the repository to test your understanding, not just your ability to run code.

### Example Questions

**1. Why should a dataset normally be split into training and testing data?**

**2. What is data leakage?**

**3. When would accuracy be insufficient as an evaluation metric?**

**4. What is the difference between precision and recall?**

**5. Why can increasing the depth of a decision tree cause overfitting?**

**6. What does `k` represent in k-NN?**

**7. What is the purpose of feature scaling?**

**8. What is the difference between supervised and unsupervised learning?**

**9. How does k-Means determine clusters?**

**10. Why is reproducibility important in machine learning?**

---

## ⚠️ Common Mistakes

Avoid these common problems when building ML models:

```text
❌ Training and testing on the same data
❌ Data leakage
❌ Ignoring missing values
❌ Using inappropriate metrics
❌ Comparing models without a consistent evaluation procedure
❌ Overfitting the training data
❌ Randomly changing parameters without justification
❌ Reporting metrics without interpretation
❌ Creating notebooks that cannot be reproduced
❌ Failing to document limitations
```

A good ML project should answer three questions:

> **What did you do?**

> **Why did you do it?**

> **What do the results actually mean?**

---

## 📚 Recommended Resources

### Books

1. **Aurélien Géron** — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
2. **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani & Jonathan Taylor** — *An Introduction to Statistical Learning with Applications in Python*
3. **Andreas C. Müller & Sarah Guido** — *Introduction to Machine Learning with Python*

These are the principal references specified for the course.

### Online Resources

* [Scikit-Learn Documentation](https://scikit-learn.org/)
* [Google Colab](https://colab.research.google.com/)
* [Kaggle](https://www.kaggle.com/)
* [Python Documentation](https://docs.python.org/3/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [NumPy Documentation](https://numpy.org/doc/)

---

## 📈 Assessment Structure

The course uses continuous practical and project-based assessment.

| Component                                  |   Weight |
| ------------------------------------------ | -------: |
| Weekly Laboratory Tasks & Coding Exercises |  **10%** |
| Notebook Checks & Project Progress         |  **10%** |
| Midterm Mini-Project                       |  **30%** |
| Final Capstone Project                     |  **30%** |
| Final Presentation & Viva                  |  **20%** |
| **Total**                                  | **100%** |

The assessment structure follows the course assessment plan.

---

## 🎓 Learning Philosophy

This repository follows a **learn → implement → evaluate → explain → improve** approach.

```text
        ┌───────────────┐
        │    LEARN      │
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │  IMPLEMENT    │
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │   EVALUATE    │
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │    EXPLAIN    │
        └───────┬───────┘
                ↓
        ┌───────────────┐
        │    IMPROVE    │
        └───────────────┘
```

The goal is not simply to produce a working model.

The goal is to understand **why the model works, how well it works, where it fails, and how to communicate the result clearly.**

---

## 🤝 Collaboration

For group projects:

* Divide responsibilities clearly.
* Use Git/GitHub for version control.
* Keep notebooks organised.
* Document important decisions.
* Review each other's code.
* Ensure every member understands their contribution.
* Keep datasets and generated files organised.

---

## 📌 Repository Guidelines

When contributing notebooks or code:

1. Use meaningful filenames.
2. Keep notebooks organised from top to bottom.
3. Explain important decisions.
4. Avoid unnecessary duplicated code.
5. Set random seeds where appropriate.
6. Record important parameters.
7. Include evaluation metrics.
8. Visualise important results.
9. Discuss limitations.
10. Make experiments reproducible.

---

## 🌟 What You Will Build

By the end of the course, you will have experience building projects that follow a complete workflow:

```text
                 MACHINE LEARNING PROJECT
                          │
          ┌───────────────┴───────────────┐
          │                               │
     Problem Definition              Dataset
          │                               │
          └───────────────┬───────────────┘
                          ↓
                   Data Preparation
                          ↓
                  Exploratory Analysis
                          ↓
                   Model Selection
                          ↓
                  Model Development
                          ↓
                  Model Evaluation
                          ↓
                   Model Comparison
                          ↓
                Reproducible Pipeline
                          ↓
                 Results & Discussion
                          ↓
                  Report + Presentation
```

---

## 🔖 Course Status

**Status:** Active / Educational Repository
**Level:** Undergraduate
**Focus:** Practical Machine Learning
**Primary Language:** Python
**Environment:** Jupyter / Google Colab
**Core Framework:** Scikit-Learn

---

## ⭐ Contribute & Learn

If you find an issue, improve an example, or develop a useful extension:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test your code.
5. Commit your changes.
6. Open a pull request.

```bash
git checkout -b feature/improve-example
git add .
git commit -m "Improve ML example"
git push origin feature/improve-example
```

---

<p align="center">
  <strong>Learn Machine Learning by building it.</strong>
</p>

<p align="center">
  Python • Data • Models • Evaluation • Projects
</p>
