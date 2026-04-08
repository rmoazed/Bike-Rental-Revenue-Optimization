# Bike-Rental-Revenue-Optimization
This project builds an end-to-end machine learning pipeline for predicting bike demand and subsequent adjusted demand, and then optimizes revenue by finding a best price at which to rent out bikes.

## Project Overview


As stated above this project builds an end-to-end machine learning pipeline to predict daily bike rental demand to predict daily bike rental demand and then extends the model to include a pricing simulation layer to optimize revenue. The model combines predictive modeling with assumptions based on customer sensitivity to price (elasticity).


## Business Problem


Companies often need to forecast demand to make operational decisions, one of which is pricing with the objective of maximizing revenue. Real-world datasets are useful for predicting baseline demand, but also often lack pricing data, which leads to the need for customer behavior simulation, as demonstrated in this project. The goal was to not only accurately predict demand, but then adjust demand based on price shift from a set base price, and evaluate the impact of this price shift on revenue. 


## Dataset


The dataset used in this project was provided by UC Irvine, and is linked here: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset


It includes a variety of input features, including temporal (year, month, weekday), categorical (season, weather conditions), and environmental (temperature, humidity, windspeed) data. The size of the dataset is about 731 observations. The target variable was total rent count on a given day (cnt).


## Data Preprocessing & Feature Engineering

Elements included in data preprocessing and feature engineering are as follows:


- Dropping of redundant columns (e.g. date, since year, month, and day are already features)
- Converting categorical variables using one-hot encoding (season, weekday, weather, etc.)
- Created cyclical features for month (sine and cosine transformations to better capture seasonality since, for example, December and January are numerically far apart but seasonally very close)
- Ensured no data leakage between train and test sets
- Split data into train and test sets


## Model Development


The model used Random Forest Regressor as it handles nonlinear relationships well, is robust to feature scaling, and is easily interpretable. Parameters such as maximum depth and minimum samples per leaf were tuned to reduce overfitting, and the model was evaluated using RSME score. 


## Model Performance


Train RSME was ~512 while test RSME was ~739, with a target mean of ~4500. This means the model generalizes well (demonstrated by the small gap between train and test RSME), and that prediction error is ~16-17% of mean demand. Performance was strong for this dataset.


## Pricing Simulation

The bike dataset, while robust, did not include any information pertaining to bike rental pricing. The project therefore introduced a simulated pricing model that assumed linear demand elasticity, implying that demand is inversely proportional to price. After a baseline demand was determined from the machine learning model, an adjusted demand was calculated as a function of price, relying on a set base price and a price sensitivity coefficient simulating how strongly customers would respond to price change. Revenue was then calculated as (price)x(adjusted demand). 


## Pricing Optimization


For each predicted demand value:

- A range of candidate prices were evaluated
- Adjusted/expected demand was computed for each candidate price
- Revenue was calculated for each candidate price

The price that maximized revenue was then selected. 


## Key Results


With a realistic elasticity assumption (alpha = 0.15), the average revenue increase was ~1.24%. Lower elasticity coefficients, such as alpha = .08, produced unrealistic behavior, always maximizing price but at the cost of dismissing the reality of customer reaction to high prices. These results demonstrate the importance of modeling assumptions when it comes to pricing decisions and, even more importantly, balancing realism with optimization. A revenue increase of 1.24% might sound underwhelming, but in a real-world scenario can actually produce meaningful increases in revenue. 


## Key Insights


- Feature engineering significantly impacts model performance
- Improper encoding can lead to unstable, misleading predictions
- Small modeling assumptions (such as elasticity) dramatically affect outcome
- Realistic models often produce modest but meaningful improvements

## Future Improvements


Potential future improvements for this project could include:


- Learning elasticity from data instead of assuming it/manually assigning value
- Incorporating time-series models
- Adding uncertainty estimates


## Folder Structure:


- Data: contains CSV file used for project
- Notebooks: contains Jupyter Notebook associated with project
- src: contains .py files
  - preprocessing.py: holds data loading, feature engineering, train/inference column alignment
  - pricing.py: holds elasticity assumption, revenue math, price optimization
  - train_bike_demand.py: trains the Random Forest, prints RSME/baseline metrics, runs pricing on test predictions, saves model artifacts
  - predict_demand.py: loads saved model and feature columns, processes new input, returns demand + pricing recommendation
 

## Installation:

pip install -r requirements.txt

