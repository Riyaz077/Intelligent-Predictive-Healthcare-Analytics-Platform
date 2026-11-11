# Notebook workflow

Create or export the following notebooks as you develop the experiments:

1. `01_Data_Collection.ipynb` — load each dataset and document its source.
2. `02_EDA_and_Visualization.ipynb` — schema, missingness, target balance and distributions.
3. `03_Data_Preprocessing.ipynb` — define target mapping and preprocessing decisions.
4. `04_Feature_Engineering.ipynb` — justified feature transformations.
5. `05_Model_Training.ipynb` — train candidate classifiers.
6. `06_Model_Evaluation.ipynb` — metrics, confusion matrices and ROC curves.
7. `07_Explainable_AI.ipynb` — permutation importance and optional SHAP analysis.

The reusable implementation is in `src/`; notebooks should call these modules rather than duplicate large amounts of code.
