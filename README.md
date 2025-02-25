# ML-Model-Sentiment-Analysis
ML-Model-Sentiment-Analysis
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
