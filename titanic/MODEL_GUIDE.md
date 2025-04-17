# Titanic Survival Prediction Model: Technical Reference Guide

This document serves as a comprehensive reference guide for the Titanic survival prediction model implementation. Use this guide to quickly look up information about the model's components, features, and implementation details when answering questions.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Feature Engineering](#feature-engineering)
3. [Feature Selection](#feature-selection)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Prediction Generation](#prediction-generation)
8. [FAQ](#faq)

## Data Preprocessing

### Dataset Loading
- Training data loaded from `train.csv`
- Test data loaded from `test.csv`
- Datasets combined for consistent preprocessing

### Missing Value Handling
- **Age**: Imputed based on Title, Pclass, and Sex group medians
- **Fare**: Imputed with median by Pclass
- **Embarked**: Filled with most common value

### Categorical Encoding
- LabelEncoder used for: Sex, Embarked, Title, Deck, FamilyGroup, AgeGroup

### Feature Scaling
- StandardScaler applied to: Age, Fare, FarePerPerson

## Feature Engineering

### Name-based Features
- **Title**: Extracted from passenger names using regex pattern `r' ([A-Za-z]+)\.'`
- Titles grouped into categories: Mr, Miss, Mrs, Master, Officer, Royalty, Other

### Family-based Features
- **FamilySize**: SibSp + Parch + 1 (includes the passenger)
- **FamilyGroup**: Categorized as 'Alone' (1), 'Small' (2-4), 'Large' (5-11)

### Cabin-based Features
- **Deck**: First letter of Cabin (A-G, U for unknown)

### Ticket-based Features
- **TicketFrequency**: Count of passengers sharing the same ticket number

### Age-based Features
- **AgeGroup**: Categorized as 'Child' (0-12), 'Teenager' (12-18), 'YoungAdult' (18-35), 'Adult' (35-60), 'Senior' (60+)

### Fare-based Features
- **FarePerPerson**: Fare divided by FamilySize

## Feature Selection

### Method
- SelectFromModel from sklearn.feature_selection
- Uses Random Forest for feature importance evaluation
- Threshold set to 'median' to select features with above-median importance

### Selected Features
- Typically includes: Sex, Title, Fare, FarePerPerson, Age, Pclass, Deck (may vary based on model runs)

## Model Architecture

### Ensemble Model Components
1. **Random Forest Classifier**
   - Optimized with GridSearchCV
   - Weight in ensemble: 2

2. **Gradient Boosting Classifier**
   - Optimized with GridSearchCV
   - Weight in ensemble: 2

3. **Extra Trees Classifier**
   - Fixed parameters: n_estimators=300, max_depth=10
   - Weight in ensemble: 1

### Voting Strategy
- Soft voting (uses probability estimates)
- Weighted voting based on model performance

## Hyperparameter Optimization

### Random Forest Parameters
- **n_estimators**: [200, 300, 400]
- **max_depth**: [7, 10, 15, None]
- **min_samples_split**: [2, 4, 6]
- **min_samples_leaf**: [1, 2, 3]
- **bootstrap**: [True, False]
- **class_weight**: [None, 'balanced']

### Gradient Boosting Parameters
- **n_estimators**: [100, 200, 300]
- **learning_rate**: [0.01, 0.05, 0.1]
- **max_depth**: [3, 4, 5, 6]
- **min_samples_split**: [2, 4, 6]
- **min_samples_leaf**: [1, 2, 3]
- **subsample**: [0.8, 0.9, 1.0]

### Cross-Validation
- StratifiedKFold with 5 splits
- Scoring metrics: accuracy, f1, precision, recall
- Refit criterion: accuracy

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Reporting
- Cross-validation mean and standard deviation for each metric
- Feature importance ranking from Random Forest model

## Prediction Generation

### Process
1. Final ensemble model trained on full training data with selected features
2. Predictions generated for test data
3. Submission file created with PassengerId and Survived columns
4. Output saved to 'titanic_submission.csv'

## FAQ

### What are the most important features for predicting survival?

Based on the Random Forest model, the most important features typically are:
1. Sex (gender)
2. Title (extracted from name)
3. Fare
4. FarePerPerson
5. Age

These features have consistently shown the highest predictive power for survival outcomes.

### How does the model handle missing values?

The model uses a sophisticated approach for handling missing values:
- **Age**: Missing values are imputed based on the median age of passengers with the same Title, Pclass, and Sex
- **Fare**: Missing values are imputed with the median fare for the same Pclass
- **Embarked**: Missing values are filled with the most common embarkation port
- **Cabin**: Missing values are marked with 'U' for unknown deck

### What ensemble method is used and why?

The model uses a VotingClassifier with soft voting (probability-based) that combines:
1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Extra Trees Classifier

This ensemble approach leverages the strengths of different algorithms to improve prediction accuracy and reduce overfitting. The models are weighted (2:2:1) to give more influence to the better-performing models.

### How is feature selection performed?

Feature selection uses SelectFromModel with a Random Forest classifier to identify the most predictive features. The threshold is set to 'median', meaning only features with importance above the median are selected. This reduces dimensionality and helps prevent overfitting.

### What cross-validation strategy is used?

The model uses StratifiedKFold with 5 splits to maintain the same class distribution in each fold. Multiple metrics (accuracy, precision, recall, F1) are calculated during cross-validation to provide a comprehensive evaluation of model performance.

### How are categorical variables encoded?

Categorical variables are encoded using LabelEncoder, which transforms categorical values into numeric indices. This is applied to Sex, Embarked, Title, Deck, FamilyGroup, and AgeGroup features.

### What is the purpose of the FarePerPerson feature?

FarePerPerson (Fare divided by FamilySize) provides a normalized measure of the ticket cost per individual, which can be more informative than the raw Fare value. This helps account for group bookings where the total fare was split among family members.

### How are age groups determined?

Age is categorized into five groups:
- Child: 0-12 years
- Teenager: 12-18 years
- YoungAdult: 18-35 years
- Adult: 35-60 years
- Senior: 60+ years

These groupings capture different life stages that may have influenced survival probability.

### What is the significance of the Title feature?

The Title feature (extracted from passenger names) captures social status and gender information that strongly correlates with survival patterns. Titles are grouped into categories (Mr, Miss, Mrs, Master, Officer, Royalty, Other) to reduce dimensionality while preserving meaningful distinctions.

### How does the model achieve its accuracy?

The model achieves high accuracy through:
1. Comprehensive feature engineering that creates informative predictors
2. Feature selection to focus on the most predictive variables
3. Ensemble learning that combines multiple algorithms
4. Hyperparameter optimization to fine-tune each model
5. Cross-validation to ensure robust performance evaluation

This multi-faceted approach addresses the complexities of the Titanic survival prediction task.