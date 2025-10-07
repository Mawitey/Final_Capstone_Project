
# USA Real Estate Price Prediction — Capstone Project
Here is link to the notebook  - https://github.com/Mawitey/Final_Capstone_Project/blob/main/USA_REAL_ESTATE_PREDICTION.ipynb

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
- Removed missing rows for clean, consistent input.
- Encoded categorical variables (LabelEncoder for high-cardinality columns, OneHotEncoder for smaller ones).
- Standardized numerical features for balanced model training.
- Split dataset: 80% training / 20% testing.

#### Exploratory Analysis
The dataset was explored visually and statistically to understand distributions and relationships among variables.
- **Histograms:** showed data distributions for price, bed, bath, and house size.
- **Correlation heatmap:** revealed strong positive relationships between house size, number of bedrooms, and price.
- **Scatter plots:** illustrated clear upward trends — as house size and number of bedrooms increased, price tended to rise.


### Results
#### Models Used
| Model | Description | Accuracy | Notes |
|--------|--------------|-----------|--------|
| Logistic Regression | Baseline linear classifier | **55.7%** | Simple and interpretable baseline. |
| Decision Tree (GridSearchCV) | Tuned for depth and split criteria | **71.2%** | Best-performing model overall. |
| K-Nearest Neighbors (KNN) | Tested with k = [3,5,7] | **60.8%** | Moderate accuracy, slower for large data. |
| Support Vector Machine (SVM) | Simplified randomized search | **57.7%** | Balanced but computationally heavy. |
| Neural Network (Keras/TensorFlow) | 1 hidden layer (100 ReLU units) | **65.8%** | Captured complex patterns effectively. |

#### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|------------|---------|-----------|
| Logistic Regression | 0.557 | 0.57 | 0.56 | 0.54 |
| Decision Tree | 0.712 | 0.72 | 0.71 | 0.71 |
| KNN | 0.608 | 0.61 | 0.61 | 0.61 |
| SVM | 0.577 | 0.58 | 0.58 | 0.57 |
| Neural Network | 0.658 | — | — | — |


- Decision Tree achieved the highest accuracy (71%) and balanced precision/recall.
- KNN and Neural Network performed moderately well.
- SVM achieved 57.7% accuracy but was optimized for speed.
- The Neural Network accuracy (65.8%) shows that even a simple deep learning model can generalize well on structured data.


### Findings
- House size, bedrooms, and bathrooms are the most influential features affecting price.
- Decision Tree is the best-performing algorithm for this dataset — combining interpretability and accuracy.
- Simplified model tuning (using smaller parameter grids and fewer folds) drastically reduced computation time without losing much accuracy.
- Neural Networks matched KNN and SVM in accuracy, proving that even simple architectures can capture meaningful patterns.
- Clean, well-scaled data is critical — dropping missing values and standardizing features improved model reliability.

### Next steps
- Tune hyperparameters for Decision Trees and Neural Networks.
- Add more details about location, such as neighborhood or nearby schools, region, distance to city center, or average income.
- Make the model available through a web app where users can enter property details and get a price category instantly.
