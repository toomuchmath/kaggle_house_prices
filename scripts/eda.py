import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import os

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

print("shape: train_df {}; test_df {}".format(train_df.shape, test_df.shape))
# shape: train_df (1460, 81); test_df (1459, 80)

train_df.info()
test_df.info()

train_df["SalePrice"].describe()

# histogram
mu, sigma = norm.fit(train_df['SalePrice'])
fig1 = plt.figure(figsize=(10, 8))
sns.distplot(train_df["SalePrice"], fit=norm)
plt.ylabel("Frequency")
plt.title("SalePrice distribution")
plt.legend([r'Normal distribution with $\mu$: {:.1f}, $\sigma$: {:.1f}'.format(mu, sigma)], loc='best')

# QQ plot
fig2 = plt.figure()
res = stats.probplot(train_df["SalePrice"], plot=plt)

# skewness and kurtosis
print("Skewness: %f" % train_df["SalePrice"].skew())
# Skewness: 1.882876
print("Kurtosis: %f" % train_df["SalePrice"].kurt())
# Kurtosis: 6.536282

# correlation matrix (only for numeric columns)
corr_matrix = train_df.corr()
fig3, ax3 = plt.subplots(figsize=(10, 8))
overview_hm = sns.heatmap(corr_matrix, cmap="YlGnBu", vmax=.8, square=True, ax=ax3)
plt.title("Correlation Matrix Overview ")

# saleprice correlation matrix
k = 10
cols = corr_matrix.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(train_df[cols].values.T)
fig4, ax4 = plt.subplots()
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values, ax=ax4)
plt.title("Correlation Matrix: Top 10")

# box plot overallqual/saleprice
fig5 = plt.figure(figsize=(8, 6))
fig5 = sns.boxplot(x=train_df["OverallQual"], y=train_df["SalePrice"])
fig5.axis(ymin=0, ymax=800000)
fig5.set_title("Overall Quality Boxplot")

# scatter plot GrLivArea/saleprice
fig6 = plt.figure()
fig6 = sns.scatterplot(x=train_df["GrLivArea"], y=train_df["SalePrice"])
fig6.axis(ymin=0, ymax=800000)
fig6.set_title("(Above) Ground Living Area scatter plot")

# missing data in train_df
total = train_df.isnull().sum().sort_values(ascending=False)
ratio = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, ratio], axis=1, keys=['Total', 'Ratio'])
missing_data.head(20)

# missing data in test_df
total = test_df.isnull().sum().sort_values(ascending=False)
ratio = (test_df.isnull().sum() / test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, ratio], axis=1, keys=['Total', 'Ratio'])
missing_data.head(40)

# looking at skewness of each numerical features
numerical_cols = train_df.dtypes[train_df.dtypes != "object"].index

skewness = train_df[numerical_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skew_df = pd.DataFrame({'Skewness': skewness})
skew_df.drop(skew_df[abs(skew_df.Skewness) < 0.75].index, inplace=True)
print(skew_df)

plt.show()
