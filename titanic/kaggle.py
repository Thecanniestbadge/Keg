# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:33:42 2025

Enhanced Titanic Survival Prediction Model with Feature Engineering
and Advanced Machine Learning Techniques

@author: Canniestbadge
"""

# This script implements an advanced machine learning solution for the Titanic survival prediction
# challenge on Kaggle. It includes comprehensive feature engineering, ensemble modeling,
# and hyperparameter optimization to achieve high prediction accuracy.

# Improved Python Script for Kaggle Titanic Competition with Advanced Feature Engineering
# and Ensemble Learning for Higher Accuracy

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Store PassengerId for submission file
test_passenger_id = test_df['PassengerId']

# Combine datasets for consistent preprocessing
print("Combining datasets for preprocessing...")
full_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# Feature Engineering
print("Performing feature engineering...")

# 1. Extract titles from names
def extract_title(name):
    """Extract title from passenger name"""
    # Fixed escape sequence in regex pattern
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

# Extract titles and create a new feature
full_df['Title'] = full_df['Name'].apply(extract_title)

# Group rare titles
print("Grouping rare titles...")
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Lady': 'Royalty',
    'Countess': 'Royalty',
    'Jonkheer': 'Royalty',
    'Capt': 'Officer',
    'Ms': 'Mrs',
    'Dona': 'Royalty'
}

# Map titles to grouped categories
full_df['Title'] = full_df['Title'].map(lambda x: title_mapping.get(x, 'Other'))

# 2. Create family size feature
print("Creating family size features...")
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1  # +1 includes the passenger

# Create family group categories
full_df['FamilyGroup'] = pd.cut(full_df['FamilySize'], 
                               bins=[0, 1, 4, 11],
                               labels=['Alone', 'Small', 'Large'])

# 3. Extract information from Cabin
print("Extracting cabin information...")
# Extract cabin deck (first letter)
full_df['Deck'] = full_df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')

# 4. Create a feature for ticket frequency
print("Creating ticket frequency feature...")
ticket_counts = full_df['Ticket'].value_counts()
full_df['TicketFrequency'] = full_df['Ticket'].map(ticket_counts)

# 5. Create age groups
print("Creating age groups...")
# First, impute missing ages based on Title, Pclass, and Sex
age_imputer = SimpleImputer(strategy='median')

# Group by these features and calculate median age
age_medians = full_df.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()

# Print percentage of missing age values before imputation
missing_age_pct = full_df['Age'].isnull().mean() * 100
print(f"Missing age values: {missing_age_pct:.2f}%")

# Fill missing ages with the corresponding group median
for i, row in full_df.iterrows():
    if pd.isna(row['Age']):
        median_age = age_medians.get((row['Title'], row['Pclass'], row['Sex']))
        if pd.notna(median_age):
            full_df.at[i, 'Age'] = median_age
        else:
            # If no matching group, use overall median
            full_df.at[i, 'Age'] = full_df['Age'].median()

# Create age groups
full_df['AgeGroup'] = pd.cut(full_df['Age'], 
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teenager', 'YoungAdult', 'Adult', 'Senior'])

# Handling missing values for other features
print("Handling remaining missing values...")
# Fare - impute with median by Pclass
for pclass in [1, 2, 3]:
    fare_median = full_df[full_df['Pclass'] == pclass]['Fare'].median()
    full_df.loc[(full_df['Fare'].isnull()) & (full_df['Pclass'] == pclass), 'Fare'] = fare_median

# Embarked - fill with most common value
full_df['Embarked'].fillna(full_df['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables
print("Encoding categorical variables...")
encoder_sex = LabelEncoder()
encoder_embarked = LabelEncoder()
encoder_title = LabelEncoder()
encoder_deck = LabelEncoder()
encoder_family_group = LabelEncoder()
encoder_age_group = LabelEncoder()

full_df['Sex'] = encoder_sex.fit_transform(full_df['Sex'])
full_df['Embarked'] = encoder_embarked.fit_transform(full_df['Embarked'])
full_df['Title'] = encoder_title.fit_transform(full_df['Title'])
full_df['Deck'] = encoder_deck.fit_transform(full_df['Deck'])
full_df['FamilyGroup'] = encoder_family_group.fit_transform(full_df['FamilyGroup'])
full_df['AgeGroup'] = encoder_age_group.fit_transform(full_df['AgeGroup'])

# Create fare per person feature
print("Creating fare per person feature...")
full_df['FarePerPerson'] = full_df['Fare'] / full_df['FamilySize']

# Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
scaled_features = ['Age', 'Fare', 'FarePerPerson']
full_df[scaled_features] = scaler.fit_transform(full_df[scaled_features])

# Select features for model training
print("Selecting features for model training...")
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
           'Title', 'FamilySize', 'Deck', 'TicketFrequency', 'FarePerPerson',
           'AgeGroup', 'FamilyGroup']

# Drop unnecessary columns
full_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Split data back to train and test
print("Splitting data back to train and test sets...")
train_processed = full_df[:len(train_df)]
test_processed = full_df[len(train_df):]

# Prepare training data
X_train = train_processed[features]
y_train = train_processed['Survived']
X_test = test_processed[features]

# Feature importance analysis with Random Forest
print("Analyzing feature importance...")
initial_rf = RandomForestClassifier(n_estimators=100, random_state=42)
initial_rf.fit(X_train, y_train)

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': initial_rf.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Top 5 most important features:")
print(feature_importance.head(5))

# Feature selection using SelectFromModel
print("Performing feature selection...")
from sklearn.feature_selection import SelectFromModel

# Use Random Forest for feature selection
feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
feature_selector.fit(X_train, y_train)

# Get selected features
selected_features_mask = feature_selector.get_support()
selected_features = [feature for feature, selected in zip(features, selected_features_mask) if selected]

print(f"Selected {len(selected_features)} features out of {len(features)}:")
print(selected_features)

# Update training and test data with selected features
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

# Create an ensemble of models
print("Building ensemble model...")

# 1. Random Forest with hyperparameter tuning
rf_param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [7, 10, 15, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'f1', 'precision', 'recall'],
    refit='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit on selected features
print("Fitting Random Forest with selected features...")
rf_grid_search.fit(X_train_selected, y_train)
rf_best = rf_grid_search.best_estimator_

print(f"Best Random Forest Parameters: {rf_grid_search.best_params_}")
print(f"Random Forest CV Accuracy: {rf_grid_search.best_score_:.4f}")

# 2. Gradient Boosting Classifier with expanded parameter grid
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0]
}

gb_grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['accuracy', 'f1', 'precision', 'recall'],
    refit='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit on selected features
print("Fitting Gradient Boosting with selected features...")
gb_grid_search.fit(X_train_selected, y_train)
gb_best = gb_grid_search.best_estimator_

print(f"Best Gradient Boosting Parameters: {gb_grid_search.best_params_}")
print(f"Gradient Boosting CV Accuracy: {gb_grid_search.best_score_:.4f}")

# 3. XGBoost Classifier
print("Adding XGBoost to the ensemble...")
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees Classifier
et_clf = ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42)
et_clf.fit(X_train_selected, y_train)

# 4. Create Voting Classifier (ensemble) with three models
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('gb', gb_best),
        ('et', et_clf)
    ],
    voting='soft',  # Use probability estimates for voting
    weights=[2, 2, 1]  # Give more weight to the best performing models
)

# Evaluate ensemble with cross-validation using multiple metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

from sklearn.model_selection import cross_validate
ensemble_cv_results = cross_validate(
    ensemble, 
    X_train_selected, 
    y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=scoring
)

# Extract accuracy scores for backward compatibility
ensemble_cv_scores = ensemble_cv_results['test_accuracy']

print(f"Ensemble Cross-validation Accuracy: {ensemble_cv_scores.mean():.4f} ± {ensemble_cv_scores.std():.4f}")

# Print all metrics
print("Ensemble cross-validation results:")
for metric, scores in ensemble_cv_results.items():
    if metric.startswith('test_'):
        metric_name = metric.replace('test_', '')
        print(f"{metric_name.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

# Train the final ensemble model on the full training data with selected features
print("Training final ensemble model...")
ensemble.fit(X_train_selected, y_train)

# Generate predictions using selected features
print("Generating predictions...")
predictions = ensemble.predict(X_test_selected)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': predictions
})

submission.to_csv('titanic_submission.csv', index=False)
print('Submission file "titanic_submission.csv" created successfully.')
print(f"Final model accuracy (cross-validation): {ensemble_cv_scores.mean():.4f}")

# Print feature importance from the Random Forest model
if hasattr(rf_best, 'feature_importances_'):
    print("\nTop 10 most important features from Random Forest:")
    rf_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_best.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    for i, row in rf_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

print("\nModel training and prediction completed successfully!")