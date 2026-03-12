RANDOM_SEED = 42

N_ACCOUNTS = 200
N_CLUSTERS = 20
SIMULATION_DAYS = 400

DECAY_LAMBDA = 0.03  # λ for exponential baseline weighting (longer memory = more lag)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}
