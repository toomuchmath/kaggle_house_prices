from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

train_df = pd.read_csv("data/train_df.csv")
train_y = pd.read_csv("data/train_y.csv")
test_df = pd.read_csv("data/test_df.csv")


def get_rmse(model, n_fold):

    k_fold = KFold(n_fold, shuffle=True, random_state=1).get_n_splits(train_df.values)
    rmse_scorer = make_scorer(mean_squared_error(squared=False))
    rmse = cross_val_score(model, train_df.values, train_y, cv=k_fold, scoring=rmse_scorer)

    return rmse


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
