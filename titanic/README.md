# Enhanced Titanic Survival Prediction Model

This project implements an advanced machine learning solution for the Titanic survival prediction challenge on Kaggle. It includes comprehensive feature engineering, ensemble modeling, and hyperparameter optimization to achieve high prediction accuracy.

## Features

- **Advanced Feature Engineering**: Extracts titles from names, creates family size features, extracts cabin information, and more.
- **Feature Selection**: Uses SelectFromModel to identify the most predictive features.
- **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Extra Trees classifiers for improved prediction accuracy.
- **Hyperparameter Optimization**: Uses GridSearchCV to find optimal parameters for each model.
- **Cross-Validation**: Evaluates model performance using multiple metrics (accuracy, precision, recall, F1).

## Requirements

All required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

Run the model with:

```bash
python kaggle.py
```

This will:
1. Load and preprocess the training and test datasets
2. Perform feature engineering
3. Train an ensemble model
4. Generate predictions
5. Create a submission file (`titanic_submission.csv`)

## Model Performance

The ensemble model achieves approximately 84.4% accuracy on cross-validation, with balanced precision and recall metrics.

## Key Features by Importance

The most important features for prediction (based on Random Forest):
1. Sex
2. Title
3. Fare
4. FarePerPerson
5. Age
6. Pclass
7. Deck

## Files

- `kaggle.py`: Main script for data processing and model training
- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `titanic_submission.csv`: Generated predictions for submission
- `requirements.txt`: Required Python packages