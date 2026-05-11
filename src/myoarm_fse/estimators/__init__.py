"""State estimators (Phase 3 fixed-gain, learned, particle, ...)."""

from myoarm_fse.estimators.base import Estimator
from myoarm_fse.estimators.fixed_kalman import (
    EstimationResult,
    FixedGainKalmanEstimator,
    aggregate_estimation_metrics,
    evaluate_estimator_on_log,
    synth_observations,
)
from myoarm_fse.estimators.learned import (
    GainPredictor,
    LearnedGainKalmanEstimator,
    LearnedGainTrainConfig,
    StateAwareGainPredictor,
    StateAwareLearnedGainKalmanEstimator,
    StateAwareTrainConfig,
    train_state_aware_gain_predictor,
    load_learned_gain_model,
    load_oracle_table,
    make_learned_gain_model_id,
    save_learned_gain_model,
    train_gain_predictor,
)

__all__ = [
    # base
    "Estimator",
    # fixed-gain
    "EstimationResult",
    "FixedGainKalmanEstimator",
    "aggregate_estimation_metrics",
    "evaluate_estimator_on_log",
    "synth_observations",
    # learned-gain (Stage A)
    "GainPredictor",
    "LearnedGainKalmanEstimator",
    "LearnedGainTrainConfig",
    "load_learned_gain_model",
    "load_oracle_table",
    "make_learned_gain_model_id",
    "save_learned_gain_model",
    "train_gain_predictor",
    # learned-gain (Stage B: state-aware)
    "StateAwareGainPredictor",
    "StateAwareLearnedGainKalmanEstimator",
    "StateAwareTrainConfig",
    "train_state_aware_gain_predictor",
]
