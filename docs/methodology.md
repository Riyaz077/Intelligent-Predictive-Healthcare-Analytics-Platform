# Methodology

## Experimental protocol
- Inspect dataset schema, missingness, duplicates and target balance.
- Define the prediction target from the dataset documentation.
- Separate features and target before splitting.
- Use a stratified train/test split when class counts permit.
- Fit imputation, encoding and scaling only on training data through a pipeline.
- Compare Logistic Regression, Random Forest, Extra Trees, Gradient Boosting and SVM; XGBoost is included when installed.
- Report accuracy, weighted precision, weighted recall, weighted F1 and ROC-AUC when supported.
- Retain the test split for final reporting; use cross-validation on training data for model selection if extending the project.

## Reproducibility
Record source URL, dataset version, preprocessing choices, target mapping, random seed, package versions and date of experiment. Avoid tuning repeatedly against the test set.
