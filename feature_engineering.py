import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.preprocessing import PowerTransformer

# import train and test data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# save ID column and drop it as it's not needed for prediction
train_id = train_df["Id"]
test_id = test_df["Id"]

train_df.drop(columns="Id", inplace=True)
test_df.drop(columns="Id", inplace=True)

# deleting outliers
train_df.drop(train_df[(train_df.GrLivArea > 4000)
                       & (train_df.SalePrice < 300000)].index, inplace=True)

# transform SalePrice using log
train_df["SalePrice"] = np.log(train_df["SalePrice"])

# visualise the transformed SalePrice distribution
sns.distplot(train_df["SalePrice"], fit=norm)
plt.ylabel("Frequency")
plt.title("SalePrice distribution")

# as well as the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df.SalePrice, plot=plt)
plt.show()

# save SalePrice as train_y and drop it as we want to have metrics columns only for our model
train_y = train_df["SalePrice"]
train_df.drop(columns="SalePrice", inplace=True)

# dealing with missing values
none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond",
             "GarageType", "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "BsmtCond", "BsmtQual", "MasVnrType"]

zero_cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1",
             "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageArea", "GarageCars"]

mode_cols = ["Electrical", "MSZoning", "Functional", "Exterior1st", "Exterior2nd",
             "SaleType", "KitchenQual"]

drop_cols = ["Utilities"]

for col in none_cols:
    train_df[col] = train_df[col].fillna("None")
    test_df[col] = test_df[col].fillna("None")

for col in zero_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

for col in mode_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# Dropping Utilities column because
train_df.drop(columns="Utilities", inplace=True)
test_df.drop(columns="Utilities", inplace=True)

# Filled LotFrontage na values using median of houses in the same neighbourhood
train_df["LotFrontage"] = train_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
test_df["LotFrontage"] = test_df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# make sure that there are no null data in both train and test datasets
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

print(train_missing[train_missing != 0], test_missing[test_missing != 0])

# convert numbers to string to indicate it's a categorical column
cat_col = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]

for col in cat_col:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

numerical_cols = train_df.dtypes[train_df.dtypes != "object"].index

skewness = train_df[numerical_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_df = pd.DataFrame({'Skewness': skewness})

skew_df.drop(skew_df[abs(skew_df.Skewness) < 0.75].index, inplace=True)
print(skew_df)

skewed_cols = skew_df.index

pt = PowerTransformer(standardize=False)
pt.fit(train_df[skewed_cols])
print(pt.lambdas_)
train_df[skewed_cols] = pd.DataFrame(pt.transform(train_df[skewed_cols]))

print(train_df[skewed_cols].skew())

