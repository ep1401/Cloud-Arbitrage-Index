# src/cai_model/model.py
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import RANDOM_STATE

def make_model():
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=5,
        max_leaf_nodes=31,
        learning_rate=0.03,
        max_iter=400,
        min_samples_leaf=100,
        l2_regularization=1e-2,
        random_state=RANDOM_STATE,
    )
