# Science Reach 2025: comparing ML models for fraud detection

Our Science Reach 2025 project. We compared how well different machine
learning models detect fraud in financial transactions, using recall and
overall accuracy.

## What we did

1. Got a synthetic financial transactions dataset from Kaggle, then cleaned
   and preprocessed it.
2. Trained and compared a range of models in DataRobot (ensembles, neural
   networks, and others), scoring each with its confusion matrix and
   checking the results with confidence intervals.
3. Built our own model with LightGBM, tuned with Optuna and trained with early
   stopping to limit overfitting. That model is the notebook in this repo.

The LightGBM model reached 99.98% accuracy. Fraud datasets are usually very imbalanced, so
accuracy on its own overstates how good a model is. Recall on the fraud class
is the number that matters more.

A possible next step is trying quantum machine learning methods for this
problem.

## Files

- `Fraud_Detection_Model.ipynb`: the LightGBM model (runs in Colab)
