## House Prices Prediction

This is based on a beginners' kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

In [eda.py](https://github.com/toomuchmath/house_prices/blob/master/scripts/eda.py), I set to explore
the raw data:
1. I had a look at the SalePrice of my train data. It exhibits positive skewness, 
longer-than-a-normal-distribution tail as well as peak at around 110,000.
![SalePrice_histogram](./figures/SalePrice_histogram.png)


In [featureengineering.py](https://github.com/toomuchmath/house_prices/blob/master/scripts/feature_engineering.py), 
my plans were as follows:
1. Remove `Id` and delete outliers
2. Log transform `SalePrice` and set it aside for later use
3. Deal with missing values
4. Convert numerical columns (which are meant to be categorical) to categorical columns
5. Use box-cox transformation (in particular, I used [powertransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
in sklearn) to ensure normality (and hence removing heteroscedasticity)
6. Get dummies
![cm](./figures/Correlation_heatmap.png)
![Top10corr](./figures/Top10corr_heatmap.png)
![GrLivArea_scatterplot](./figures/GrLivArea_scatterplot.png)
![OverallQual_boxplot](./figures/OverallQual_boxplot.png)

![SalePrice_qqplot](./figures/SalePrice_qqplot.png)
