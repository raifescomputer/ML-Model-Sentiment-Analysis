# Sentiment Analysis with Machine Learning

This repository contains a simple machine learning model for sentiment analysis, trained on the IMDB movie reviews dataset.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Model Persistence](#model-persistence)
- [Contributing](#contributing)
- [License](#license)

## Project Description

This project aims to build a sentiment analysis model that can classify movie reviews as either positive or negative. The model is trained using the Naive Bayes algorithm and utilizes TF-IDF vectorization for text preprocessing.

## Dataset

The model is trained and evaluated on the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset contains 50,000 movie reviews, each labeled with a sentiment (positive or negative).

The dataset file `IMDB Dataset.csv` should be placed in the `data/` directory.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Install the required dependencies using pip:

```bash
pip install -r requirements.txt
Installation
Clone the repository:

Bash

1. git clone [https://github.com/yourusername/ML-Model-Sentiment-Analysis.git](https://github.com/yourusername/ML-Model-Sentiment-Analysis.git)
cd ML-Model-Sentiment-Analysis
(Optional) Create a virtual environment:

Bash

2. python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
Install dependencies:

Bash

3. pip install -r requirements.txt
Place the IMDB Dataset.csv file in the data/ directory.

4. Place the IMDB Dataset.csv file in the data/ directory.

Usage
To run the sentiment analysis script:

Bash

python sentiment_model.py
This will train the model, evaluate its performance, and print the results, including:

Cross-validation scores
Accuracy on the test set
Classification report (precision, recall, F1-score)
Confusion matrix
Prediction for a new review
Model Training
The model is trained using the Multinomial Naive Bayes algorithm. Text preprocessing includes:

TF-IDF vectorization
Stop word removal
The dataset is split into training and testing sets with an 80/20 ratio.

## Evaluation
The model's performance is evaluated using:

## Accuracy
Classification report (precision, recall, F1-score)
Confusion matrix
Cross-validation (5-fold)
The cross-validation scores and test set accuracy should be around 86-87%.

## Model Persistence
To save the trained model:

Python

import joblib
joblib.dump(model, 'sentiment_model.joblib')
To load the saved model:

Python

import joblib
loaded_model = joblib.load('sentiment_model.joblib')
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.   

