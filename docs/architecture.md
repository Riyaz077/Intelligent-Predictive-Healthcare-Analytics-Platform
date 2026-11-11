# System Architecture

1. **Input:** User supplies a de-identified, permitted CSV dataset.
2. **Validation:** The application checks the selected target and available features.
3. **Preprocessing:** Numeric values are median-imputed and scaled; categorical values are mode-imputed and one-hot encoded. Transformations are fitted inside a scikit-learn pipeline to reduce train/test leakage.
4. **Training:** Candidate classifiers are trained independently for the selected dataset.
5. **Evaluation:** Held-out metrics and confusion matrices are generated.
6. **Interpretability:** Permutation importance estimates how much each input feature affects a selected scoring metric.
7. **Presentation:** Streamlit displays data summaries, model comparisons and demonstration predictions.

The architecture is intended for reproducible educational experiments, not clinical deployment.
