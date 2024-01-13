from numpy import linspace


FIT_CV_SPLITTING_STRATEGY = 3
FIT_CV_VERBOSE = 3
LEARNING_CURVE_SPLITTING_STRATEGY_N_SPLITS = 5
LEARNING_CURVE_SPLITTING_STRATEGY_TEST_SIZE = 0.2
LEARNING_CURVE_SPLITTING_STRATEGY_TRAIN_SIZES = [0.01, 0.05]
LEARNING_CURVE_SPLITTING_STRATEGY_TRAIN_SIZES += [*linspace(0.2, 1., 5)]
LEARNING_CURVE_VERBOSE = 3
N_JOBS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.15
