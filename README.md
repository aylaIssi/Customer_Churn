# Telco Customer Churn Prediction

## Description

This project involves data science and machine learning techniques to predict customer churn for a telecom company. Customer churn refers to the likelihood of a customer leaving a product or service. The project uses various classification algorithms to predict customer churn cases for a telecom company.

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost


## Requirements

To run the project, the following libraries are required. You can install the required libraries using the following command in your terminal or command prompt:

```
pip install -r requirements.txt

```
## Usage

Loading the dataset and performing basic data analysis:

```
df = load()
check_dataframe(df)

```
### Data cleaning and preprocessing steps:

```
# Handling missing and outlier values
df.dropna(inplace=True)
df.drop(columns='customerID', inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Creating new features
df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
df.loc[(df["Contract"] == "Month-to-month"), "Subscription"] = "No"
df.loc[((df["Contract"] == "One year") | (df["Contract"] == "Two years")), "Subscription"] = "Yes"
df["LoF_Tenure"] = pd.qcut(df["tenure"], 5, labels=["L1", "L2", "L3", "L4", "L5"])
df["LoF_MonthlyCharges"] = pd.qcut(df["MonthlyCharges"], 5, labels=["L1", "L2", "L3", "L4", "L5"])
df["LoF_TotalCharges"] = pd.qcut(df["TotalCharges"], 5, labels=["L1", "L2", "L3", "L4", "L5"])

# Encoding processes
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)
df = one_hot_encoder(df, cat_cols, drop_first=True)

# Standardization for numerical variables
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


```

### Modeling and evaluation:

```
# Random Forests model and hyperparameter optimization
rf_model = RandomForestClassifier()
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)

# Similar steps can be applied for other classification algorithms (Gradient Boosting, XGBoost, LightGBM, CatBoost).


```

### Model Performance and Visualization of Important Features:

```
# Visualizing model performance
plot_importance(rf_model, X)

# Drawing accuracy curves for hyperparameter optimization
val_curve_params(rf_model, X, y, param_name="n_estimators", param_range


```
