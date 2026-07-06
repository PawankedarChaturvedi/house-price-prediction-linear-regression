# house-price-prediction-linear-regression
# 🏡 House Price Prediction using Linear Regression (From Scratch)

## 📌 Project Overview

This project demonstrates how Linear Regression works by implementing the complete learning algorithm **from scratch using NumPy**, without relying on machine learning libraries such as Scikit-Learn for model training.

The objective is to predict house prices based on a single feature:

- Carpet Area (sq. ft.)

using Gradient Descent optimization.

---

# 📖 Problem Statement

Predict the selling price of a house using its carpet area.

Given the carpet area of a property, train a Linear Regression model to estimate the expected house price while minimizing the prediction error using Gradient Descent.

---

# 📊 Dataset

Dataset Source:

Housing Price Prediction Dataset (Kaggle)

Feature Used:

- Carpet Area

Target Variable:

- Price (Lakhs)

The dataset is loaded directly from Kaggle using KaggleHub.

---

# 🧠 Machine Learning Concepts Covered

- Linear Regression
- Cost Function
- Mean Squared Error (MSE)
- Gradient Descent
- Model Training
- Prediction
- Model Evaluation
- R² Score

---

# 📐 Mathematical Model

Linear Regression Equation

f(x)=wx+b

Where

- x = Carpet Area
- w = Weight
- b = Bias
- f(x) = Predicted House Price

---

# 📉 Cost Function

The model minimizes the Mean Squared Error Cost Function.

\[
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f(x_i)-y_i)^2
\]

Where

- m = Number of training examples
- y = Actual price
- f(x) = Predicted price

The objective is to find the values of **w** and **b** that minimize this cost.

---

# ⚙️ Gradient Descent

The model updates the parameters using Gradient Descent.

Weight Update

\[
w=w-\alpha\frac{\partial J}{\partial w}
\]

Bias Update

\[
b=b-\alpha\frac{\partial J}{\partial b}
\]

Where

- α = Learning Rate

Training continues until the cost converges.

---

# 🚀 Workflow

1. Load dataset from Kaggle
2. Select Carpet Area as input feature
3. Split data into Train/Test sets
4. Implement Linear Regression from scratch
5. Compute Cost Function
6. Compute Gradients
7. Train using Gradient Descent
8. Predict house prices
9. Evaluate using R² Score

---

# 📈 Results

Model Evaluation

- Algorithm: Linear Regression
- Training Method: Gradient Descent
- Learning Rate: 1e-7
- Iterations: 10,000

Performance

single_Feature_Dataset_Model_Building = single_Feature_Dataset_All.loc[: len(single_Feature_Dataset_All)/2]
- R² Score: **0.3122**

single_Feature_Dataset_Model_Building = single_Feature_Dataset_All.loc[: len(single_Feature_Dataset_All)/4]
- R² Score: **0.5056** 


The model is able to capture the general trend between carpet area and house prices. Since only one feature is used, the predictive performance is limited.

---

# 📊 Visualizations

The notebook includes:

- Cost vs Iterations
- Predicted House Prices

<img width="732" height="442" alt="image" src="https://github.com/user-attachments/assets/afe61aac-942f-4e51-86b8-bcf90c00dd7e" />

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- KaggleHub
- Google Colab

---

# 📁 Project Structure

```
House-Price-Prediction/

│── House_Price_Prediction.ipynb
│── README.md
│── requirements.txt
│── images/
```

---

# 🔮 Future Improvements

- Use Multiple Linear Regression
- Feature Scaling
- Data Cleaning
- Handle Missing Values
- Feature Engineering
- Compare with Scikit-Learn Implementation
- Add Train/Test Cross Validation
- Improve R² Score using multiple features
- Deploy using Streamlit
- Build REST API using FastAPI

---

# 🎯 Learning Outcome

This project helped me understand

- How Linear Regression works internally
- How Gradient Descent updates model parameters
- Why Cost Function decreases during training
- How predictions are generated
- How machine learning models are trained without external ML libraries

---

## ⭐ Author

Pawan Chaturvedi

Machine Learning Journey (Andrew Ng ML Specialization)
