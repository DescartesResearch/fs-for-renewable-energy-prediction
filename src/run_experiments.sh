#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Global knobs (override via env, e.g. HPO_MAX_ITER=200 ./experiments.sh)
# -----------------------------
HPO_MAX_ITER=${HPO_MAX_ITER:-25}
HPO_TRAIN_TIME_LIMIT=${HPO_TRAIN_TIME_LIMIT:-60}
WARMUP_MAX_ITER=${WARMUP_MAX_ITER:-25}
N_BOOTSTRAP_SAMPLES=${N_BOOTSTRAP_SAMPLES:-100}

RANDOM_SEED=${RANDOM_SEED:-27}
N_JOBS=${N_JOBS:-2}
OVERWRITE=${OVERWRITE:-false}

# -----------------------------
# Common args shared by all runs
# -----------------------------
COMMON_ARGS=(
  --hpo_train_time_limit "$HPO_TRAIN_TIME_LIMIT"
  --hpo_max_iter "$HPO_MAX_ITER"
  --hpo_early_stop
  --warm_starts
  --warmup_max_iter "$WARMUP_MAX_ITER"
  --warmup_early_stop
  --bootstrapping
  --n_bootstrap_samples "$N_BOOTSTRAP_SAMPLES"
  --n_jobs "$N_JOBS"
  --random_seed "$RANDOM_SEED"
)
if [[ "$OVERWRITE" == "true" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

# -----------------------------
# Helper to run a single experiment
# usage: run_exp <name> [extra args...]
# -----------------------------
run_exp() {
  local name="$1"; shift
  echo ""
  echo "========== Running experiment: ${name} =========="
  set -x
  uv run ./main.py --name "$name" "${COMMON_ARGS[@]}" "$@"
  { set +x; } 2>/dev/null
}

# -----------------------------
# Define experiments
# Keep names unique to avoid overwriting artifacts/logs.
# -----------------------------
declare -a EXP_NAMES
declare -a EXP_ARGS

# CSFS and SFS experiments:

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_csfs-feature_importance-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method feature_importance --group_size 3")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_csfs-correlation_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method correlation")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_csfs-random-gs3_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method random --group_size 3")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_csfs-singletons_per_feature_set")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --hpo_mode per_feature_set --fs_method CSFS --fast_mode --direction backward --clustering_method singletons")

# Baseline experiments

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_mlp_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model mlp --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_xgboost_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model xgboost --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_lgbm_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model lgbm --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("wind-T11_rf_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain wind --asset_id T11 --model rf --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_mlp_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model mlp --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_xgboost_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model xgboost --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_lgbm_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model lgbm --features forecast_available --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-2_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-3_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-5_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-8_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-10_digital_twin_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features digital_twin --n_features 10 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-2_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 2 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-3_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 3 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-5_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 5 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-8_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 8 --fs_method RF_FI")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_mutual_info")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --fs_method mutual_info")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_f_value")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --fs_method f_value")

EXP_NAMES+=("pv-01_rf_n-10_forecast_available_RF_FI")
EXP_ARGS+=("--domain pv --asset_id 01 --model rf --features forecast_available --n_features 10 --fs_method RF_FI")

# Total number of experiments: 560



# -----------------------------
# Helpers for CLI parsing
# -----------------------------
usage() {
  local max_i=$(( ${#EXP_NAMES[@]} - 1 ))
  echo "Usage:"
  echo "  $(basename "$0")              # run all experiments sequentially"
  echo "  $(basename "$0") list         # list experiments with indices"
  echo "  $(basename "$0") <index>      # run a single experiment by index (0..$max_i)"
  echo "  $(basename "$0") <i-j>        # run a range of experiments inclusive (e.g. 3-5)"
}

in_range_or_die() {
  local idx="$1"
  local max_i=$(( ${#EXP_NAMES[@]} - 1 ))
  if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
    echo "Invalid index: $idx" >&2; usage; exit 2
  fi
  if (( idx < 0 || idx > max_i )); then
    echo "Index $idx out of range (0..$max_i)." >&2; usage; exit 2
  fi
}

run_indices() {
  for i in "$@"; do
    in_range_or_die "$i"
    # shellcheck disable=SC2086
    run_exp "${EXP_NAMES[$i]}" ${EXP_ARGS[$i]}
  done
}

# -----------------------------
# CLI
# -----------------------------
if [[ "${1:-}" == "list" ]]; then
  for i in "${!EXP_NAMES[@]}"; do
    printf "%2d: %s  |  %s\n" "$i" "${EXP_NAMES[$i]}" "${EXP_ARGS[$i]}"
  done
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage; exit 2
fi

if [[ $# -eq 1 ]]; then
  arg="$1"
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    run_indices "$arg"
    echo ""; echo "Done."
    exit 0
  elif [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    in_range_or_die "$start"
    in_range_or_die "$end"
    if (( start > end )); then
      echo "Range start ($start) must be <= end ($end)." >&2; exit 2
    fi
    to_run=()
    for ((i=start; i<=end; i++)); do
      to_run+=("$i")
    done
    run_indices "${to_run[@]}"
    echo ""; echo "Done."
    exit 0
  else
    usage; exit 2
  fi
fi

# Default: run all sequentially
for i in "${!EXP_NAMES[@]}"; do
  # shellcheck disable=SC2086
  run_exp "${EXP_NAMES[$i]}" ${EXP_ARGS[$i]}
done

echo ""
echo "All experiments finished."