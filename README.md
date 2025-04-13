## Pythagoras (?)
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
- [Usage](#usage)
  - [Generating the Dataset](#generating-the-dataset)
  - [Training the Models](#training-the-models)
  - [Evaluating the Models](#evaluating-the-models)
- [Results](#results)
- [Author](#author)
- [Ackwnowledgment](#acknowledgment)

## Overview

The goal of this project is to predict the hypotenuse \(c\) of a right-angled triangle given its sides \(a\) and \(b\). It uses both a traditional Linear Regression approach and an MLP Regressor to illustrate the differences in performance when dealing with non-linear relationships. Visualizations of the predictions versus actual values are generated and saved as images.

## Usage

    Generating the dataset:

python src/generate_dataset.py

Training the models:

    Linear Regression:

python src/train.py

MLP Regressor:

    python src/mlp_train.py

Evaluating the models:

    Linear Regression:

python src/evaluate.py

MLP Regressor:

        python src/mlp_evaluate.py

## Results

    Linear Regression: Limited by its linear nature when approximating the non-linear hypotenuse equation.

    MLP Regressor: Effectively captures the non-linear relationship with low error.
    
## Authors

    Created by elicolds

## Acknowledgment

    Teacher: @Maritaganta

Software III – Robotic Fabrication SeminarMRAC01 – IAAC
