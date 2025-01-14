# Customer Churn Prediction for DTH Services

## Objective
The project aims to develop a machine learning model to predict customer churn for a Direct-to-Home (DTH) service provider. By identifying potential churners, the company can implement targeted strategies to retain customers and minimize revenue loss. The focus is to:
- Build a churn prediction model using historical customer data.
- Provide actionable insights and business recommendations to reduce churn.
- Enhance customer retention through segmented campaigns and offers.

---

## Tools and Technologies
- **Programming Language**: Python (Jupyter Notebook)
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Techniques**: 
  - Decision Trees, Random Forests, Logistic Regression
  - K-Nearest Neighbors (KNN), Naïve Bayes, Linear Discriminant Analysis (LDA)
  - Artificial Neural Networks (ANN), Ensemble Methods (Boosting, Bagging)
- **Data Handling**: Feature Selection, SMOTE for data balancing, Variable Transformation
- **Visualization Tools**: Matplotlib and Seaborn
- **Documentation**: PDF Report and Presentation

---

## Project Workflow

### 1. Problem Statement and Business Understanding
DTH services face intense competition from cable operators and OTT platforms, leading to high customer churn rates. The company’s goal is to reduce churn by understanding key factors influencing customer behavior and predicting potential churners to implement re-engagement strategies.

### 2. Data Cleaning and Preprocessing
- Handled missing values using median (numerical) and mode (categorical).
- Treated outliers using the IQR method.
- Performed dummy variable transformation for categorical variables.

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Identified key predictors like tenure, cashback, and complaints.
- **Bivariate Analysis**: Analyzed relationships between variables and churn.
- **Multivariate Analysis**: Assessed feature interactions and correlations to refine feature selection.

### 4. Machine Learning Models
The following models were implemented and evaluated:
1. **Base Models**: Decision Tree, Random Forest, KNN, Logistic Regression, Naïve Bayes, LDA, ANN
2. **Tuned Models**: Hyperparameter optimization was performed.
3. **SMOTE Models**: Addressed class imbalance to improve minority class prediction.

### Performance Metrics
- **Evaluation Criteria**: Accuracy, Precision, Recall, F1-score, and AUC.
- **Key Observations**:
  - **Best Models**: Random Forest, KNN, Bagging (highest F1-scores).
  - SMOTE enhanced model performance for imbalanced datasets.
  - Precision and Recall trade-offs were optimized to focus on actionable insights for churn prevention.

### Key Metrics: Precision, Recall, and F1-Score
- **Accuracy**: Measures overall correctness but may mislead for imbalanced datasets.
- **Precision**: Focuses on reducing False Positives.
- **Recall**: Focuses on reducing False Negatives.
- **F1-Score**: Balances Precision and Recall; suitable for scenarios where both false positives and false negatives carry significant costs.

### Why F1-Score?
In this project:
- High recall is crucial to identify churners and prevent revenue loss.
- High precision ensures marketing resources are not wasted on non-churners.
- F1-score balances both, making it the most relevant metric for actionable insights.

---

### 5. Performance Summary

### Insights from Base Models
- Tree-based models (Decision Tree, Random Forest) and ensemble methods outperformed other models.
- Models like ANN, Logistic Regression, and Naïve Bayes struggled due to the imbalanced dataset.

---

### Hyperparameter Tuning
- Random Forest, KNN, and Gradient Boost models improved significantly with tuning.
- Logistic Regression, ANN, and LDA showed marginal improvements but still underperformed compared to tree-based models.

---

### Data Balancing with SMOTE
After applying SMOTE:
- Recall for minority classes improved for Random Forest, KNN, and ensemble methods.
- F1-scores increased, validating the impact of data balancing.
- ANN and Logistic Regression saw minimal gains, highlighting their limitations for this dataset.

---
## 6. Results

## Interpretation of the Most Optimum Model and Its Business Implications

## Train Set Results After Hyperparameter Tuning

| Model               | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
|---------------------|----------|---------------|------------|--------------|---------------|------------|--------------|
| CART                | 0.92     | 0.95          | 0.97       | 0.96         | 0.81          | 0.72       | 0.76         |
| Random Forest       | 0.93     | 0.94          | 0.98       | 0.96         | 0.88          | 0.69       | 0.77         |
| ANN                 | 0.90     | 0.92          | 0.97       | 0.94         | 0.82          | 0.56       | 0.66         |
| Logistic Regression | 0.88     | 0.90          | 0.97       | 0.93         | 0.75          | 0.45       | 0.56         |
| LDA                 | 0.88     | 0.89          | 0.97       | 0.93         | 0.75          | 0.41       | 0.53         |
| Naïve Bayes         | 0.87     | 0.90          | 0.94       | 0.92         | 0.62          | 0.51       | 0.56         |
| KNN                 | **1.00** | **1.00**      | **1.00**   | **1.00**     | **1.00**      | **1.00**   | **1.00**     |
| Ada Boost           | **1.00** | **1.00**      | **1.00**   | **1.00**     | **1.00**      | **1.00**   | **1.00**     |
| Gradient Boost      | **1.00** | **1.00**      | **1.00**   | **1.00**     | **1.00**      | **1.00**   | **1.00**     |
| Bagging             | **1.00** | **1.00**      | **1.00**   | **1.00**     | **1.00**      | **1.00**   | **1.00**     |

---

## Test Set Results After Hyperparameter Tuning

| Model               | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
|---------------------|----------|---------------|------------|--------------|---------------|------------|--------------|
| CART                | 0.90     | 0.93          | 0.95       | 0.94         | 0.74          | 0.65       | 0.69         |
| Random Forest       | 0.91     | 0.93          | 0.97       | 0.95         | 0.82          | 0.63       | 0.71         |
| ANN                 | 0.90     | 0.91          | 0.97       | 0.94         | 0.80          | 0.53       | 0.65         |
| Logistic Regression | 0.88     | 0.90          | 0.97       | 0.93         | 0.74          | 0.45       | 0.56         |
| LDA                 | 0.88     | 0.89          | 0.97       | 0.93         | 0.75          | 0.43       | 0.54         |
| Naïve Bayes         | 0.86     | 0.91          | 0.93       | 0.92         | 0.60          | 0.52       | 0.56         |
| KNN                 | **0.97** | **0.98**      | **0.99**   | **0.98**     | **0.94**      | **0.89**   | **0.92**     |
| Ada Boost           | 0.95     | 0.97          | 0.97       | 0.97         | 0.85          | 0.86       | 0.86         |
| Gradient Boost      | 0.96     | 0.97          | 0.99       | 0.98         | 0.94          | 0.84       | 0.89         |
| Bagging             | 0.95     | 0.95          | 0.99       | 0.97         | 0.95          | 0.72       | 0.82         |

---

## Key Insights and Interpretation

### **KNN Performance**:
- KNN achieved a **test F1-score of 0.92** and an **accuracy of 0.97**, making it the best-performing model in terms of overall metrics.
- Despite its simplicity, KNN outperformed complex algorithms due to effective hyperparameter tuning and data preprocessing.

### **Why KNN Performed Better**:
- **Instance-based Learning**: KNN uses the nearest neighbors for prediction, making it ideal for datasets with non-linear relationships.
- **Scaling Impact**: Proper feature scaling enhanced KNN’s ability to compute meaningful distances between data points.
- **Optimal Hyperparameters**: Fine-tuning parameters such as `n_neighbors` and `weights` helped balance bias and variance.

### **Business Implications**:
- **High Recall**: Accurately identifies churners, allowing proactive retention strategies.
- **Balanced Precision**: Reduces false positives, optimizing marketing costs.
- **Ease of Deployment**: KNN’s interpretability makes it practical for real-time churn prediction.

## 7. Conclusion:
Sometimes, simpler models like KNN outperform advanced algorithms. In this case, KNN provided an optimal balance of precision and recall, making it highly effective for predicting customer churn and driving actionable business insights.

### Strategies and Recommendations
- **Customer Segmentation**: Prioritize resolving complaints for high-risk churners.
- **Personalized Campaigns**: Offer rewards for long-tenure customers.
- **Feedback Loops**: Use social media analytics for proactive engagement.
- **Enhanced Service**: Simplify customer complaint processes and improve response times.

---

### Insights and Recommendations
- **Key Predictors**: Tenure, complaints in the last year, cashback usage, and customer service interactions.
- **Business Strategies**:
  1. Target churners with personalized offers (e.g., discounts, rewards).
  2. Focus on resolving complaints promptly.
  3. Enhance customer experience through guided assistance and educational content.
  4. Use social media feedback and text mining for proactive engagement.

---

### Features of the Project
- Comprehensive EDA with actionable insights.
- Rigorous preprocessing ensuring data quality.
- Multiple machine learning models tested and optimized.
- Business recommendations based on data-driven insights.

---

### Strengths
1. **End-to-End Workflow**: Covers the entire data science pipeline—from problem formulation to actionable recommendations.
2. **Model Variety**: Application of multiple algorithms of machine learning techniques.
3. **Business Impact**: Clear connection between data insights and business strategies align technical solutions with organizational goals.

### Future Improvements
1. **Explainability**: Adding SHAP or LIME for interpretability of model predictions to enhance the project’s depth.
2. **Automation**: Include scripts for automating the data pipeline and model training.
3. **Visualization**: Add more interactive visuals (e.g., dashboards) for presenting results effectively.

---


