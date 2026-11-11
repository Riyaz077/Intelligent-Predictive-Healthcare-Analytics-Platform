import pandas as pd
from src.preprocessing import split_features_target, build_preprocessor

def test_split_features_target():
    df = pd.DataFrame({"age": [20, 30, 40], "target": [0, 1, 0]})
    X, y = split_features_target(df, "target")
    assert list(X.columns) == ["age"]
    assert len(y) == 3

def test_preprocessor_builds():
    X = pd.DataFrame({"age": [20, None, 40], "sex": ["F", "M", "F"]})
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 3
