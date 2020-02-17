import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import os


train_df = pd.read_csv(os.getcwd() + r"\input\train.csv")


def overview(df):
    print("shape:", df.shape)
    print("-"*40)
    print("info:")
    print(df.info())
    print("-" * 40)
    print("columns:", df.columns)


overview(train_df)

