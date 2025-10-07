
# USA Real Estate Price Prediction — Capstone Project
Here is link to the notebook  - https://github.com/Mawitey/Final_Capstone_Project/blob/main/Final_Capstone_Project.ipynb

###  Problem statement
The goal of this project is to predict home price categories in the U.S. real estate market — Low, Medium, High, or Luxury — based on property features such as house size, bedrooms, bathrooms, and location.

With more than 1 million listings, the dataset offers insight into how these features affect price.
By building and comparing several machine learning models, the project identifies which approach most accurately predicts price category while remaining efficient and interpretable.

#### Data Overview
Source: Realtor.com dataset (1,084,909 rows).
Target variable: price_category derived from price values:
- **Low:** < $200,000  
- **Medium:** $200,000 – $500,000  
- **High:** $500,000 – $1,000,000  
- **Luxury:** > $1,000,000 
Main features: bedrooms, bathrooms, lot size, house size, city, and state.

#### Data Preparation
Removed missing rows for clean, consistent input.
Encoded categorical variables (LabelEncoder for high-cardinality columns, OneHotEncoder for smaller ones).
Standardized numerical features for balanced model training.
Split dataset: 80% training / 20% testing.

#### Exploratory Analysis
Histograms showed data distributions for price, bed, bath, and house size.
Correlation analysis revealed strong positive relationships between house size, number of bedrooms, and price.
Outliers confirmed the wide range of home prices across different states.


### Results
#### Models Used
| Model | Description | Accuracy | Notes |
|--------|--------------|-----------|--------|
| Logistic Regression | Baseline linear classifier | **59%** | Simple and interpretable baseline. |
| Decision Tree (GridSearchCV) | Tuned for depth and split criteria | **72%** | Best-performing model overall. |
| K-Nearest Neighbors (KNN) | Tested with k = [3,5,7] | **64%** | Moderate accuracy, slower for large data. |
| Support Vector Machine (SVM) | Simplified randomized search | **60%** | Balanced but computationally heavy. |
| Neural Network (Keras/TensorFlow) | 1 hidden layer (100 ReLU units) | **63%** | Captured complex patterns effectively. |

#### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|------------|---------|-----------|
| Decision Tree | 0.720 | 0.722 | 0.720 | 0.716 |
| KNN | 0.644 | 0.643 | 0.644 | 0.642 |
| SVM | 0.596 | 0.596 | 0.596 | 0.581 |
| Neural Network | 0.633 | — | — | — |


Decision Tree achieved the highest accuracy (≈72%) and balanced precision/recall.
KNN and Neural Network performed moderately well.
SVM achieved 60% accuracy but was optimized for speed.
The Neural Network accuracy (63%) shows that even a simple deep learning model can generalize well on structured data.


### Findings
House size, bedrooms, and bathrooms are the most influential features affecting price.
Decision Tree is the best-performing algorithm for this dataset — combining interpretability and accuracy.
Simplified model tuning (using smaller parameter grids and fewer folds) drastically reduced computation time without losing much accuracy.
Neural Networks matched KNN and SVM in accuracy, proving that even simple architectures can capture meaningful patterns.
Clean, well-scaled data is critical — dropping missing values and standardizing features improved model reliability.

### Next steps
Add more details about location, such as neighborhood or nearby schools.
Make the model available through a web app where users can enter property details and get a price category instantly.
