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
                       and (train_df.SalePrice < 300000)].index, inplace=True)

