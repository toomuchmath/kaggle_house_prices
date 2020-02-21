import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import lightgbm as lgb

train_df = pd.read_csv("data/train_df.csv")
test_df = pd.read_csv("data/test_df.csv")
output_df = pd.read_csv("data/sample_submission.csv")
train_y = pd.read_csv("data/train_y.csv")
train_y = np.array(train_y).ravel()


def get_rmse(model, n_fold):

    k_fold = KFold(n_fold, shuffle=True, random_state=1).get_n_splits(train_df.values)
    rmse = cross_val_score(model, train_df.values, train_y, cv=k_fold,
                           scoring="neg_root_mean_squared_error")

    return -rmse


# Lasso Regression
# Using RobustScaler() to make this model less sensitive to outliers
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# Elastic Net Regression
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=9, random_state=1))

# Kernel Ridge Regression
krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Gradient Boosting Regression
# huber loss would make the model more robust to outliers
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=1)

# XGBoost
xgb_reg = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
                           min_child_weight=2, n_estimators=2200, reg_alpha=0.4640, reg=0.8571,
                           subsample=0.5213, silent=True, random_state=1, nthread=-1)

# LightGBM
lgb_reg = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720,
                            max_bin=55, bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.2319,
                            feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6,
                            min_sum_hessian_in_leaf=11)

models = [lasso, enet, krr, gbr, xgb_reg, lgb_reg]
for model in models:
    scores = get_rmse(model, 5)
    print("{} scores: {} \n Average score: {}".format(str(model), scores, scores.mean()))

"""
# calculating scores of different models
lasso_scores = get_rmse(lasso, 5)
print("Lasso scores: {} \n Average score: {}".format(lasso_scores, lasso_scores.mean()))

enet_scores = get_rmse(enet, 5)
print("Elastic Net scores: {} \n Average score: {}".format(enet_scores, enet_scores.mean()))

krr_scores = get_rmse(krr, 5)
print("Kernel Ridge scores: {} \n Average score: {}".format(krr_scores, krr_scores.mean()))

gbr_scores = get_rmse(gbr, 5)
print("Gradient Boosting scores: {} \n Average score: {}".format(gbr_scores, gbr_scores.mean()))

xgb_scores = get_rmse(xgb_reg, 5)
print("eXtreme Gradient Boosting scores: {} \n Average score: {}".format(xgb_scores, xgb_scores.mean()))

lgb_scores = get_rmse(lgb_reg, 5)
print("Light Gradient Boosting scores: {} \n Average score: {}".format(lgb_scores, lgb_scores.mean()))
"""
estimators = [('enet', enet), ('krr', krr), ('gbr', gbr)]

stack_reg = StackingRegressor(estimators=estimators, final_estimator=lasso)
stack_reg_scores = get_rmse(stack_reg, 5)
print("Stacking Regressor scores: {} \n Average score: {}".format(stack_reg_scores, stack_reg_scores.mean()))

"""
stack_reg.fit(train_df, train_y)
output_df["SalePrice"] = stack_reg.predict(test_df)
output_df["SalePrice"] = np.expm1(output_df["SalePrice"])
output_df.to_csv("data/submission_stack_5.csv", index=False)


lgb_reg.fit(train_df, train_y)
output_df["SalePrice"] = lgb_reg.predict(test_df)
output_df["SalePrice"] = np.expm1(output_df["SalePrice"])
output_df.to_csv("data/submission_lgb.csv", index=False)
"""

