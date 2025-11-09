# UFC Predictor

A machine learning project that predicts UFC fight outcomes.  

This project uses logistic regression to predict the winner of UFC fights by analyzing differences in fighter statistics. The model is trained on historical UFC data from 1994-2025 and achieves approximately 80% accuracy.

The model is less accurate for fighters with less fights in the UFC

## Disclaimer!!

This model makes mistakes. If you use this model to bet on fights, it is 100% on you if the prediction is incorrect

## Dataset

The dataset used is publicly available on Kaggle:

[UFC Datasets 1994–2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)

The model uses UFC.csv specifically

Credit to creators Aditya Ratan Nandan K

You must download the dataset yourself and place the relevant CSV file in the `data/` folder before running the scripts.

## Features

The model uses the following statistics to make its predictions:
- Significant striking accuracy
- Strike defense
- Takedown accuracy
- Takedown defense
- Submission attempts
- Control time
- Reach
- Height
- Win percentage

## Installation

Install required dependencies:
pip install pandas numpy scikit-learn joblib

Download the UFC dataset from Kaggle and place UFC.csv in a data/ directory

## Usage

To create the trained model, run ufc_model.py and place ufc_model.pkl and scaler.pkl into a model/ directory

To make predictions, simply run app.py and follow the instructions in the terminal. The output will be the predicted winner and confidence %

## Project Structure
├── data/
│   └── ufc.csv
├── model/
│   ├── ufc_model.pkl
│   └── scaler.pkl
├── ufc_model.py
├── app.py
└── README.md

## Future Plans

This project will be deployed as a web application using Flask for the backend and React for the frontend.