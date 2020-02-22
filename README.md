## House Prices Prediction

This is based on a beginners' kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

In [featureengineering.py](https://github.com/toomuchmath/house_prices/blob/master/scripts/feature_engineering.py), 
my plans were as follows:
1. Remove `Id` and delete outliers
2. Log transform `SalePrice` and set it aside for later use
3. Deal with missing values
4. Convert numerical columns (which are meant to be categorical) to categorical columns
5. Use box-cox transformation (in particular, I used [powertransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
in sklearn) to ensure normality (and hence removing heteroscedasticity)
6. Get dummies
