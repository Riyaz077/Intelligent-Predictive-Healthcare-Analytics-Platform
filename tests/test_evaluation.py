import pandas as pd
from src.evaluation import evaluate_models

def test_evaluate_models_returns_metrics():
    X = pd.DataFrame({"x1": [0,1,2,3,4,5,6,7,8,9,10,11],
                      "x2": [1,0,1,0,1,0,1,0,1,0,1,0]})
    y = pd.Series([0,0,0,0,0,0,1,1,1,1,1,1])
    metrics, fitted = evaluate_models(X, y, test_size=0.25, random_state=7)
    assert not metrics.empty
    assert "F1_weighted" in metrics.columns
    assert len(fitted) == len(metrics)
