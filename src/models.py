"""Candidate classifiers and model pipeline construction."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from .preprocessing import build_preprocessor

def candidate_estimators(random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=random_state, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=random_state
        )
    except ImportError:
        pass
    return models

def build_model_pipelines(X, random_state=42):
    preprocessor = build_preprocessor(X)
    return {
        name: Pipeline([("preprocess", preprocessor), ("classifier", estimator)])
        for name, estimator in candidate_estimators(random_state).items()
    }
