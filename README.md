# Diabetes_Prediction


Data Exploratory Dependencies:
- `numpy`: A library for numerical operations in Python, providing support for large, multi-dimensional arrays and mathematical functions.
- `pandas`: A powerful data manipulation library that offers data structures and data analysis tools, particularly useful for handling structured data.
- `matplotlib.pyplot`: A plotting library for creating static, animated, and interactive visualizations in Python.
- `seaborn`: A data visualization library based on matplotlib, providing additional aesthetic options and statistical graphics.
- `StandardScaler` (from `sklearn.preprocessing`): A class for standardizing features by removing the mean and scaling to unit variance.
- `SMOTE` (from `imblearn.over_sampling`): A class for performing Synthetic Minority Over-sampling Technique (SMOTE), a method for oversampling the minority class to address class imbalance.

Model Building Dependencies:
- `train_test_split` (from `sklearn.model_selection`): A function for splitting the dataset into training and testing sets.
- `cross_val_score` (from `sklearn.model_selection`): A function for performing cross-validation and obtaining the model's performance scores.
- `GridSearchCV` (from `sklearn.model_selection`): A class for performing grid search over specified hyperparameters and finding the best model.
- `StratifiedKFold` (from `sklearn.model_selection`): A class for performing stratified sampling during cross-validation.
- `LogisticRegression` (from `sklearn.linear_model`): A class implementing logistic regression, a linear model for binary classification.
- `DecisionTreeClassifier` (from `sklearn.tree`): A class implementing decision tree-based classification algorithms.
- `RandomForestClassifier` (from `sklearn.ensemble`): A class implementing random forest, an ensemble of decision trees for classification.
- `AdaBoostClassifier` (from `sklearn.ensemble`): A class implementing AdaBoost, an ensemble learning method using weighted majority voting of weak classifiers.
- `KNeighborsClassifier` (from `sklearn.neighbors`): A class implementing k-nearest neighbors algorithm for classification.
- `SVC` (from `sklearn.svm`): A class implementing Support Vector Machines (SVM) for classification.
- `CatBoostClassifier` (from `catboost`): A gradient boosting framework that provides a high-performance implementation of gradient boosting for classification.
- `XGBClassifier` (from `xgboost`): An optimized gradient boosting library that offers high performance and flexibility for classification.
- `GaussianNB` (from `sklearn.naive_bayes`): A class implementing Gaussian Naive Bayes, a simple probabilistic classifier based on Bayes' theorem.

Model Evaluation Dependencies:
- `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `accuracy_score`, `classification_report` (from `sklearn.metrics`): Functions and metrics for evaluating classification models and generating performance scores and reports.

Warning Dependencies:
- `warnings`: A module for issuing and controlling warnings in Python. In this code, UserWarning and FutureWarning are filtered to ignore the respective warnings.

Dumping File:
- `pickle`: A module for serializing and deserializing Python objects, used for saving and loading model files.

