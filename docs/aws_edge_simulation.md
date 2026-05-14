# AWS Edge-Constrained Simulation

This benchmark runs the conditional face-recognition experiment in Docker with CPU and RAM limits. The correct wording is **AWS edge-constrained simulation**.

Do not report these numbers as measurements from a physical Raspberry Pi board. AWS t4g.small is ARM-based, but it is separate hardware. Docker limits provide a reproducible constrained environment for comparing risk-constrained decision methods.

## AWS EC2 Setup

Recommended instance:

```text
EC2 instance: t4g.small
OS: Ubuntu Server ARM64
Storage: 20-30 GB
```

SSH example:

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

Install dependencies:

```bash
sudo apt update
sudo apt install -y git docker.io python3 python3-pip python3-venv
sudo usermod -aG docker $USER
```

Log out and back in if Docker permission does not update. You can also test with:

```bash
docker ps
```

## Clone And Run

```bash
git clone <repo-url>
cd <repo>
chmod +x scripts/run_aws_edge_benchmark.sh
./scripts/run_aws_edge_benchmark.sh
```

The default run builds `Dockerfile.edge` and evaluates:

```text
edge_1cpu_512mb
edge_1cpu_1gb
edge_2cpu_2gb
```

Each profile runs:

```bash
python scripts/run_conditional_experiment.py \
  --dataset synthetic \
  --max-pairs 1000 \
  --output-dir outputs/aws_edge_benchmark/<profile> \
  --methods M0,M1,M4,M5,M6 \
  --far-budgets 0.01,0.02,0.03,0.05 \
  --defer-margin 0.03 \
  --calibration-split 0.5 \
  --calibration-seed 42
```

## Custom Data Run

```bash
METHODS=M0,M1,M4,M5,M6 \
FAR_BUDGETS=0.01,0.02,0.03,0.05 \
DEFER_MARGIN=0.03 \
MAX_PAIRS=1000 \
DATASET=custom \
DATA_DIR=data \
FACE_MODEL_NAME=mobilefacenet \
./scripts/run_aws_edge_benchmark.sh
```

For custom data, InsightFace and ONNX Runtime must be available in the container. If they are missing, the custom benchmark should fail clearly instead of silently using mock embeddings. Synthetic runs are useful for smoke testing only.

## Outputs

Expected directory:

```text
outputs/aws_edge_benchmark/
  edge_1cpu_512mb/
    per_sample_log.csv
    summary_by_method.csv
    summary_by_condition.csv
    summary_by_far_budget.csv
    calibration_thresholds.csv
    latency_summary.csv
    config_used.json
    aws_edge_env.json
    run.log
    plots/
  edge_1cpu_1gb/
    ...
  edge_2cpu_2gb/
    ...
  combined_edge_summary.csv
  combined_latency_summary.csv
  combined_far_budget_summary.csv
  combined_calibration_thresholds.csv
  edge_benchmark_report.md
```

## Interpretation

Use the benchmark to compare:

- `FRR_dark @ FAR <= alpha`: whether a method reduces low-light false rejection at the same accepted false-accept risk.
- `actual FAR_active`: whether the reported run respects the FAR budget on active decisions.
- `defer_rate` and `automation_rate`: whether the method is pushing too many samples to verification.
- `latency_p95` and `ram_peak_mb`: whether runtime and memory stay inside the simulated edge budget.

Write:

```text
Thresholds are calibrated under FAR budgets.
The goal is to reduce FRR at the same FAR constraint.
Deferred samples are reported explicitly and are not hidden.
```

Report the result as:

```text
AWS edge-constrained simulation under Docker CPU/RAM limits
```

Do not write:

```text
physical Raspberry Pi measurement
equivalent to a Raspberry Pi
identical to an edge board
```

Real Raspberry Pi or Jetson benchmarking should be treated as future work or optional validation.
