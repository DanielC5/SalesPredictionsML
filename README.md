# Customer Purchase Prediction with Random Forest

This project predicts customer purchase behavior (Purchase, No Purchase, or Return) using customer demographics and financial data. Multiple classification models were evaluated, with a Random Forest Classifier selected as the final model after hyperparameter tuning. The goal was to investigate why a company was experiencing declining sales.

## Features
- Predict customer purchase outcome from demographic attributes (Gender, Age, Estimated Salary).
- Preprocessing pipeline with StandardScaler and ordinal encoding built into an sklearn Pipeline.
- Compared five classification models: Linear Regression, Random Forest, Decision Tree, KNN, and SVM.
- Hyperparameter tuning with RandomizedSearchCV (up to 10,000 iterations).
- Serialized model for inference via joblib.

## Project Structure
```
SalesPredictionsML/
├── SalesForecasting.ipynb      # Full analysis, model training, and evaluation
├── SalePredictionModel.pkl     # Trained Random Forest model
└── data/
    └── DanielC.csv             # Customer dataset (10,001 records)
```

## Dataset
- **Records:** 10,001 customers
- **Features:** User ID, Name, Gender, Age, Estimated Salary
- **Target:** Purchased (1 = bought, 0 = did not buy, -1 = returned)
- **Class Distribution:** 44.9% purchased, 50.3% not purchased, 4.8% returned

## Project Workflow

1. **Data Exploration & Cleaning**:
   - Dropped non-predictive columns (User ID, Name).
   - Checked for missing values (none found).
   - Analyzed feature distributions and correlations.

2. **Preprocessing**:
   - Ordinal encoded Gender (0 = Female, 1 = Male).
   - Applied StandardScaler to Age and Estimated Salary.
   - Built reproducible sklearn Pipeline.
   - 80/20 train-test split.

3. **Feature Engineering**:
   - Explored derived features (salary per age, salary times age, age per user ID).
   - No significant improvement in correlations; kept original features.

4. **Model Selection**:
   | Model | Training Accuracy | CV F1 Score (mean) |
   |-------|------------------|--------------------|
   | Linear Regression (rounded) | 50.3% | N/A |
   | Random Forest | 99.96% | 0.4656 |
   | Decision Tree | 99.97% | 0.4490 |
   | KNN | 65.8% | 0.4704 |
   | SVM | 50.3% | 0.3387 |

5. **Hyperparameter Tuning**:
   - Used RandomizedSearchCV on the Random Forest model.
   - Tuned: `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `bootstrap`.
   - Best tuned model: 63.4% training accuracy, 0.4732 mean CV F1 score, 49% test accuracy.

6. **Inference**:
   - Trained model saved as `SalePredictionModel.pkl`.
   - Prediction function accepts customer attributes and returns purchase outcome.

## Feature Importance
- **Estimated Salary** had the highest positive impact on purchase prediction.
- **Gender** had the second highest impact.
- **Age** had negligible effect.

All feature correlations with the target were extremely weak (< 0.01), indicating purchase behavior is largely driven by factors not captured in this dataset.

## Limitations
- Severe class imbalance: returns represent only 4.8% of the data.
- Very weak feature-target correlations suggest missing key predictive variables.
- 20% accuracy gap between training and test sets indicates overfitting despite tuning.
- No stratified sampling during train-test split, potentially under-representing the return class.

## Setup Instructions
1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib
   ```
3. Run the notebook:
   ```bash
   jupyter notebook SalesForecasting.ipynb
   ```

## Conclusion
After evaluating multiple models and tuning hyperparameters, the weak correlations between available features and purchase outcomes revealed that Gender, Age, and Estimated Salary alone are insufficient predictors of customer behavior. The analysis ultimately pointed to **product quality degrading over time** as the underlying driver of declining sales — a factor not captured in the customer demographic data. Collecting product-level data (quality metrics, customer reviews, defect rates) would be necessary to build a more effective predictive model.
