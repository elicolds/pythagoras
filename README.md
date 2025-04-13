This project demonstrates how to approximate the mathematical formula for the hypotenuse of a right-angled triangle:

\[
c = \sqrt{a^2 + b^2}
\]

using shallow machine learning techniques. The project compares two approaches:
- **Linear Regression**: Which struggles with the non-linearity of the square root function.
- **MLP Regressor (Multi-Layer Perceptron)**: Which can effectively model the non-linear relationship.

This assignment is part of the Software III - MRAC01 course.

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating the Dataset](#generating-the-dataset)
  - [Training the Models](#training-the-models)
  - [Evaluating the Models](#evaluating-the-models)
- [Results](#results)
- [License](#license)

## Overview

The goal of this project is to predict the hypotenuse \(c\) of a right-angled triangle given its sides \(a\) and \(b\). It uses both a traditional Linear Regression approach and an MLP Regressor to illustrate the differences in performance when dealing with non-linear relationships. Visualizations of the predictions versus actual values are generated and saved as images.

## Folder Structure

pythagoras/ ├── data/
│ └── triangle_dataset.csv # Generated synthetic dataset ├── models/
│ ├── linear_regression_model.pkl # Trained Linear Regression model │ └── mlp_model.pkl # Trained MLPRegressor model ├── results/
│ ├── pred_vs_true.png # Visualization for Linear Regression predictions │ └── pred_vs_true_mlp.png # Visualization for MLPRegressor predictions ├── src/
│ ├── generate_dataset.py # Generates the synthetic dataset │ ├── train.py # Trains the Linear Regression model │ ├── evaluate.py # Evaluates the Linear Regression model │ ├── mlp_train.py # Trains the MLPRegressor model │ └── mlp_evaluate.py # Evaluates the MLPRegressor model ├── requirements.txt # Project dependencies └── README.md # This file


## Installation

1. **Clone the Repository (using SSH):**

   ```bash
   git clone git@github.com:YOUR_USERNAME/pythagoras.git
   cd pythagoras

    Create and Activate a Virtual Environment:

python3 -m venv venv
source venv/bin/activate

Install the Dependencies:

    pip install -r requirements.txt

Usage
Generating the Dataset

Run the following command to generate the synthetic dataset:

python src/generate_dataset.py

This script creates a CSV file (triangle_dataset.csv) in the data/ folder.
Training the Models
Linear Regression Model

Train the Linear Regression model with:

python src/train.py

The trained model is saved as linear_regression_model.pkl in the models/ folder.
MLP Regressor Model

Train the MLP Regressor model with:

python src/mlp_train.py

The trained model is saved as mlp_model.pkl in the models/ folder.
Evaluating the Models
Linear Regression Evaluation

Evaluate the performance of the Linear Regression model by running:

python src/evaluate.py

This script calculates performance metrics (MSE and R2R2) and generates a plot (pred_vs_true.png) saved in the results/ folder.
MLP Regressor Evaluation

Evaluate the MLP Regressor by running:

python src/mlp_evaluate.py

This script generates performance metrics and saves the corresponding plot (pred_vs_true_mlp.png) in the results/ folder.
Results

    Linear Regression: Exhibits higher error due to its limitations in modeling the non-linear function c=a2+b2c=a2+b2

    ​.

    MLP Regressor: Accurately captures the non-linear relationship, achieving very low Mean Squared Error and a nearly perfect R2R2 score.

Both visualizations comparing predictions to actual values are available in the results/ folder.
