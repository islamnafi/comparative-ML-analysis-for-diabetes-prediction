# Comparative ML Analysis for Diabetes Prediction

This project uses the PIMA Indians Diabetes Database to build and compare four different machine learning classification models. The primary goal is to predict the onset of diabetes (`Outcome`) based on several diagnostic features.

The entire workflow, from preprocessing to model tuning, is encapsulated in `scikit-learn` pipelines for robustness and reproducibility.

## Dataset

* [cite_start]**Source:** PIMA Indians Diabetes Database 
* [cite_start]**Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age` [cite: 16, 34]
* [cite_start]**Target:** `Outcome` (0 for No Diabetes, 1 for Diabetes) [cite: 184, 270]

## Project Workflow

1.  **Exploratory Data Analysis (EDA):**
    * Loaded the dataset using `pandas`.
    * Checked for missing values.
    * Visualized feature distributions using `matplotlib` histograms.
    * Analyzed feature correlations using a `seaborn` heatmap. `Glucose` was found to be the most correlated feature with the `Outcome`.

2.  **Preprocessing Pipeline:**
    * [cite_start]The data was split into training (80%) and testing (20%) sets.
    * A `scikit-learn` `Pipeline` was created to chain two preprocessing steps:
        * `StandardScaler()`: To scale all features.
        * `PCA()`: For dimensionality reduction.

3.  **Model Training & Hyperparameter Tuning:**
    * `GridSearchCV` was used to systematically tune hyperparameters for both the `PCA` step (`n_components`) and the specific classifier[cite: 209].
    * `StratifiedKFold` (with 5 splits) was used for cross-validation to handle the class distribution.

## Models & Results

Four different classifiers were evaluated using the same pipeline structure. The final accuracy score was measured on the held-out test set.

| Model | Best Hyperparameters | Test Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | 67.5%  |
| **K-Nearest Neighbors** | 69.5%  |
| **Random Forest** | **70.8%**  |
| **XGBoost** | 68.2%  |

### Conclusion

The **Random Forest Classifier** provided the best performance on the test set, with an accuracy of **70.8%**. Confusion matrices were also generated for each model to visualize their true positive, true negative, false positive, and false negative rates.

## How to Run

1.  Clone this repository:
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```
3.  Ensure the `diabetes.csv` file is in the same directory.
4.  Run the `diabetes_classification_pipeline.ipynb` notebook in a Jupyter environment.
