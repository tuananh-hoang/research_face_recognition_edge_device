#!/usr/bin/env bash
set -uo pipefail

IMAGE_NAME="${IMAGE_NAME:-face-edge}"
DATASET="${DATASET:-synthetic}"
DATA_DIR="${DATA_DIR:-data}"
MAX_PAIRS="${MAX_PAIRS:-1000}"
METHODS="${METHODS:-M0,M1,M4,M5,M6}"
FAR_BUDGETS="${FAR_BUDGETS:-0.01,0.02,0.03,0.05}"
DEFER_MARGIN="${DEFER_MARGIN:-0.03}"
CALIBRATION_SPLIT="${CALIBRATION_SPLIT:-0.5}"
CALIBRATION_SEED="${CALIBRATION_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_edge_benchmark}"
ROBUST_ENHANCEMENT="${ROBUST_ENHANCEMENT:-clahe}"
FACE_MODEL_NAME="${FACE_MODEL_NAME:-mobilefacenet}"
FACE_DET_SIZE="${FACE_DET_SIZE:-320,320}"

profiles=(
  "edge_1cpu_512mb|1|512m"
  "edge_1cpu_1gb|1|1g"
  "edge_2cpu_2gb|2|2g"
)

mkdir -p "${OUTPUT_ROOT}"

echo "Building Docker image for AWS edge-constrained simulation: ${IMAGE_NAME}"
if ! docker build -f Dockerfile.edge -t "${IMAGE_NAME}" .; then
  echo "Docker build failed. Fix Dockerfile.edge or dependency installation before running profiles."
  exit 1
fi

for profile_spec in "${profiles[@]}"; do
  IFS="|" read -r profile cpus memory <<< "${profile_spec}"
  profile_dir="${OUTPUT_ROOT}/${profile}"
  mkdir -p "${profile_dir}"

  cat > "${profile_dir}/profile_config.json" <<EOF
{
  "profile": "${profile}",
  "cpu_limit": "${cpus}",
  "memory_limit": "${memory}",
  "dataset": "${DATASET}",
  "data_dir": "${DATA_DIR}",
  "max_pairs": "${MAX_PAIRS}",
  "methods": "${METHODS}",
  "far_budgets": "${FAR_BUDGETS}",
  "defer_margin": "${DEFER_MARGIN}",
  "calibration_split": "${CALIBRATION_SPLIT}",
  "calibration_seed": "${CALIBRATION_SEED}",
  "face_model_name": "${FACE_MODEL_NAME}",
  "face_det_size": "${FACE_DET_SIZE}",
  "benchmark_type": "edge-constrained simulation"
}
EOF

  echo ""
  echo "============================================================"
  echo "Running ${profile} (--cpus=${cpus}, --memory=${memory})"
  echo "============================================================"

  set +e
  docker run --rm \
    --cpus="${cpus}" \
    --memory="${memory}" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/data:/app/data:ro" \
    -e FACE_MODEL_NAME="${FACE_MODEL_NAME}" \
    -e FACE_DET_SIZE="${FACE_DET_SIZE}" \
    "${IMAGE_NAME}" \
    sh -lc "python scripts/check_aws_edge_env.py --output '${profile_dir}/aws_edge_env.json' && \
      python scripts/run_conditional_experiment.py \
        --dataset '${DATASET}' \
        --data-dir '${DATA_DIR}' \
        --max-pairs '${MAX_PAIRS}' \
        --methods '${METHODS}' \
        --far-budgets '${FAR_BUDGETS}' \
        --defer-margin '${DEFER_MARGIN}' \
        --calibration-split '${CALIBRATION_SPLIT}' \
        --calibration-seed '${CALIBRATION_SEED}' \
        --face-model '${FACE_MODEL_NAME}' \
        --face-det-size '${FACE_DET_SIZE}' \
        --robust-enhancement '${ROBUST_ENHANCEMENT}' \
        --output-dir '${profile_dir}'" \
    > "${profile_dir}/run.log" 2>&1
  status=$?

  if [ "${status}" -ne 0 ]; then
    echo "Profile ${profile} failed with exit code ${status}. See ${profile_dir}/run.log"
  else
    echo "Profile ${profile} completed. See ${profile_dir}/run.log"
  fi
done

python scripts/summarize_edge_benchmark.py --input-dir "${OUTPUT_ROOT}"
