# MobileFaceNet / Edge Face Model

The project now treats `mobilefacenet` as the default face-recognition model alias for edge-oriented runs.

In InsightFace, the practical model pack used for this alias is:

```text
mobilefacenet -> buffalo_sc
```

`buffalo_sc` uses a lightweight MBF@WebFace600K recognition model and a small detector, so it is the appropriate InsightFace pack for the edge-constrained simulation baseline. This avoids switching to a server-oriented pack such as `buffalo_l`.

## Local Run

Synthetic smoke test:

```bash
python scripts/run_conditional_experiment.py \
  --dataset synthetic \
  --max-pairs 200 \
  --face-model mobilefacenet \
  --output-dir outputs/mobilefacenet_smoke
```

Custom image benchmark:

```bash
python scripts/run_conditional_experiment.py \
  --dataset custom \
  --data-dir data \
  --max-pairs 300 \
  --face-model mobilefacenet \
  --face-det-size 320,320 \
  --output-dir outputs/mobilefacenet_custom
```

If InsightFace is missing, custom runs fail clearly by default. Use `--allow-mock-embedder` only for local smoke tests, not for report results.

## AWS Edge-Constrained Simulation

```bash
DATASET=custom \
DATA_DIR=data \
MAX_PAIRS=300 \
METHODS=M0,M1,M2,M3,M4 \
FACE_MODEL_NAME=mobilefacenet \
FACE_DET_SIZE=320,320 \
./scripts/run_aws_edge_benchmark.sh
```

Terraform variable:

```hcl
face_model_name = "mobilefacenet"
face_det_size   = "320,320"
```

## Alternative Packs

You can still test other InsightFace packs:

```bash
FACE_MODEL_NAME=buffalo_s ./scripts/run_aws_edge_benchmark.sh
FACE_MODEL_NAME=buffalo_l ./scripts/run_aws_edge_benchmark.sh
```

For the report's edge framing, keep `mobilefacenet`/`buffalo_sc` as the default unless you are explicitly running an ablation.
