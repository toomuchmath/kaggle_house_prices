import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# save ID column and drop it as it's not needed for prediction
train_id = train_df["Id"]
test_id = test_df["Id"]

train_df.drop(columns = "Id", inplace = True)
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

fig = plt.figure()
res = stats.probplot(train_df.SalePrice, plot=plt)
plt.show()

# save SalePrice as train_y and drop it as we want to have metrics columns only
train_y = train_df["SalePrice"]
train_df.drop(columns="SalePrice", inplace=True)

# dealing with missing values
full_df = pd.concat([train_df, test_df], ignore_index=True)

