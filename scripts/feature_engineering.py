import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import PowerTransformer, StandardScaler

train_df = pd.read_csv("../data/train.csv")    # train_size = (1460, 81)
test_df = pd.read_csv("../data/test.csv")

# save ID column and drop it as it's not needed for prediction
train_id = train_df["Id"]
test_id = test_df["Id"]

train_df.drop(columns="Id", inplace=True)   # train_size = (1460, 80)
test_df.drop(columns="Id", inplace=True)

# deleting outliers (2 of them)
train_df.drop(train_df[(train_df.GrLivArea > 4000)
                       & (train_df.SalePrice < 300000)].index, inplace=True)    # train_size = (1458, 80)
train_df = train_df.reset_index(drop=True)

# transform SalePrice using log(1+x)
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# visualise the transformed SalePrice distribution
sns.distplot(train_df["SalePrice"], fit=norm)
plt.ylabel("Frequency")
plt.title("SalePrice distribution")

# as well as the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df.SalePrice, plot=plt)
# plt.show()

# save SalePrice as train_y and drop it as we want to have metrics columns only for our model
train_y = train_df["SalePrice"]
train_df.drop(columns="SalePrice", inplace=True)    # train_size = (1458, 79)

# dealing with missing values
none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond",
             "GarageType", "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "BsmtCond", "BsmtQual", "MasVnrType"]

zero_cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1",
             "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageArea", "GarageCars"]

mode_cols = ["Electrical", "KitchenQual"]

groupby_mode_cols = ["Functional", "Exterior1st", "Exterior2nd", "SaleType"]

for col in none_cols:
    train_df[col] = train_df[col].fillna("None")
    test_df[col] = test_df[col].fillna("None")

for col in zero_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

for col in mode_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

for col in groupby_mode_cols:
    train_df[col] = train_df.groupby("Neighborhood")[col].transform(
        lambda x: x.fillna(x.mode()[0])
    )
    test_df[col] = test_df.groupby("Neighborhood")[col].transform(
        lambda x: x.fillna(x.mode()[0])
    )

test_df["MSZoning"] = test_df.groupby("MSSubClass")["MSZoning"].transform(
    lambda x: x.fillna(x.median())
)

train_df["LotFrontage"] = train_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

test_df["LotFrontage"] = test_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# Dropping Utilities column because most rows have the same value
train_df.drop(columns="Utilities", inplace=True)    # train_size = (1458, 78)
test_df.drop(columns="Utilities", inplace=True)

# make sure that there are no null data in both train and test datasets
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

print("missing data", train_missing[train_missing != 0], test_missing[test_missing != 0])

# convert numbers to string to indicate it's a categorical column
cat_col = ["MSSubClass", "YrSold", "MoSold"]

for col in cat_col:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# add a new feature
train_df["TotalSF"] = train_df["TotalBsmtSF"] + train_df["1stFlrSF"] + train_df["2ndFlrSF"]
test_df["TotalSF"] = test_df["TotalBsmtSF"] + test_df["1stFlrSF"] + test_df["2ndFlrSF"]

numerical_cols = train_df.dtypes[train_df.dtypes != "object"].index

# transforming data into a normal distribution (to remove heteroscedasticity)
skewness = train_df[numerical_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_df = pd.DataFrame({'Skewness': skewness})

skew_df.drop(skew_df[abs(skew_df.Skewness) < 0.75].index, inplace=True)
skewed_cols = skew_df.index

# I would like to use box-cox
# but it requires that all the values to be strictly positive (> 0),
# hence I am doing a "plus one" on all data points,
# similar to what I did when performing log transformation on SalePrice
train_df[skewed_cols] = train_df[skewed_cols].add(1)
test_df[skewed_cols] = test_df[skewed_cols].add(1)

# instantiate a powertransformer and do a box-cox transformation on the skewed columns
pt = PowerTransformer(method='box-cox')
pt.fit(train_df[skewed_cols])
train_df[skewed_cols] = pd.DataFrame(pt.transform(train_df[skewed_cols]))
test_df[skewed_cols] = pd.DataFrame(pt.transform(test_df[skewed_cols]))

# get dummies
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# on inspection, test_df has fewer columns than train_df after .get_dummies()
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[train_df.columns]
print(train_df.shape, test_df.shape)    # same shape so we are ready to go!

standard_scaler = StandardScaler()
train_df = pd.DataFrame(standard_scaler.fit_transform(train_df))
test_df = pd.DataFrame(standard_scaler.transform(test_df))

train_df.to_csv("../data/train_df.csv", index=False)
test_df.to_csv("../data/test_df.csv", index=False)
train_y.to_csv("../data/train_y.csv", index=False)