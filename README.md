# Machine Learning Based Prediction of Final Batting Score in ICC Cricket World Cup Matches

## Project Overview

This project predicts the final batting score of a cricket team using
machine learning regression algorithms.

The project uses historical ICC Cricket World Cup data and predicts the
final score based on the current state of an innings.

## Problem Statement

The objective is to predict the final score of a batting team during an
innings using information such as overs completed, runs scored, wickets,
teams, city, current run rate and remaining overs.

This is a regression problem because the target variable is a continuous
numerical value.

## Dataset

The dataset contains 21,031 records.

The final model uses 13 input features:

- Year
- City
- Team 1
- Team 2
- Batting team
- Bowling team
- Over Number
- Runs Scored in over
- Runs Scored till that over
- Wicket Taken in over
- Wickets Taken till that over
- Current Run Rate
- Remaining Overs

## Feature Engineering

Two additional features were created:

### Current Run Rate

Current Run Rate is calculated as:

CRR = Runs Scored till that over / Over Number

### Remaining Overs

Remaining Overs is calculated as:

Remaining Overs = 50 - Over Number

## Train-Test Split

A group-based 80:20 train-test split was used.

Training data:

16870 records

Testing data:

4161 records

Grouping by match prevents observations from the same match from being
randomly distributed between training and testing data.

## Machine Learning Models

Five regression models were trained:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression
5. Extra Trees Regression

## Model Evaluation

The models were evaluated using:

- MAE
- MSE
- RMSE
- R²

Lower MAE, MSE and RMSE are preferred, while a higher R² is preferred.

## Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 47.26 | 3701.13 | 60.84 | 0.234 |
| Decision Tree | 69.11 | 7808.07 | 88.36 | -0.616 |
| Random Forest | 51.37 | 4457.66 | 66.77 | 0.078 |
| Gradient Boosting | 45.80 | 3447.68 | 58.72 | 0.287 |
| Extra Trees | 43.64 | 3337.32 | 57.77 | 0.309 |

Extra Trees achieved the best overall performance among the tested
models.

## Sample Prediction

Actual Final Score: 286

Predicted Final Score: 284.7

Absolute Error: 1.3 runs

## Project Structure

```text
worldcup_run_prediction/
│
├── data/
├── logs/
├── models/
├── notebooks/
├── outputs/
├── reports/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore