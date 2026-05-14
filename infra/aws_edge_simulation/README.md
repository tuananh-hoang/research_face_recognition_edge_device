# Terraform AWS Edge-Constrained Simulation

This Terraform stack lets you run the repository benchmark from your personal machine without manually creating EC2 resources.

It provisions:

- one Ubuntu ARM64 EC2 instance, default `t4g.small`;
- Docker and Git on the instance;
- an IAM role for AWS Systems Manager Session Manager;
- an optional private S3 bucket for benchmark outputs;
- user data that clones the repo, checks out `conditional-pipeline`, runs `scripts/run_aws_edge_benchmark.sh`, and uploads outputs.

This is an **AWS edge-constrained simulation**, not a physical Raspberry Pi or Jetson measurement.

## Prerequisites On Your Personal Machine

Install:

- Terraform
- AWS CLI

Configure AWS credentials with IAM Identity Center/SSO when possible:

```bash
aws configure sso
aws sso login --profile <your-profile>
```

Then export the profile:

```bash
export AWS_PROFILE=<your-profile>
```

PowerShell:

```powershell
$env:AWS_PROFILE="<your-profile>"
```

## Basic Run

From the repo root:

```bash
cd infra/aws_edge_simulation
terraform init
terraform apply
```

Terraform prints:

- EC2 instance ID;
- public IP;
- SSM session command;
- S3 result path.

Watch setup progress through Session Manager:

```bash
aws ssm start-session --target <INSTANCE_ID> --region ap-southeast-1 --profile <your-profile>
sudo tail -f /var/log/face-edge-sim/user_data.log
```

Benchmark outputs on EC2:

```text
/opt/face-edge-sim/repo/outputs/aws_edge_benchmark/
```

If S3 upload is enabled, outputs are also uploaded to:

```text
s3://<bucket>/aws-edge-benchmark/
```

The default bucket name is based on project name, AWS account ID, and region. If that bucket name already exists, set:

```bash
terraform apply -var='results_bucket_name=your-globally-unique-bucket-name'
```

## SSH Access Optional

SSM is the safer default. If you still want SSH:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/face_edge_aws
```

Find your current public IP and pass it as `/32`:

```bash
terraform apply \
  -var='public_key_path=/home/you/.ssh/face_edge_aws.pub' \
  -var='allowed_ssh_cidrs=["YOUR_PUBLIC_IP/32"]'
```

Then:

```bash
ssh -i ~/.ssh/face_edge_aws ubuntu@<PUBLIC_IP>
```

Do not commit private keys or `terraform.tfvars`.

## Custom Dataset

The repo ignores `data/`, so custom images will not automatically exist on EC2. Upload them to S3 first:

```bash
aws s3 sync data s3://<your-data-bucket>/face-data/
```

Then run:

```bash
terraform apply \
  -var='benchmark_dataset=custom' \
  -var='custom_data_s3_uri=s3://<your-data-bucket>/face-data/' \
  -var='max_pairs=300'
```

Custom runs require InsightFace/ONNX Runtime in the Docker image. If dependencies are missing, the benchmark should fail clearly rather than silently using mock embeddings.

## Useful Variables

```hcl
aws_region          = "ap-southeast-1"
aws_profile         = "your-sso-profile"
repo_branch         = "conditional-pipeline"
instance_type       = "t4g.small"
benchmark_dataset   = "synthetic"
max_pairs           = 1000
methods             = "M0,M1,M2,M3,M4"
face_model_name     = "mobilefacenet"
face_det_size       = "320,320"
shutdown_after_run  = false
```

You can place non-secret values in `terraform.tfvars`, but that file is git-ignored.

## Download Results

From S3:

```bash
aws s3 sync s3://<bucket>/aws-edge-benchmark/latest/ ./aws_edge_benchmark_results
```

Or from EC2:

```bash
scp -r ubuntu@<PUBLIC_IP>:/opt/face-edge-sim/repo/outputs/aws_edge_benchmark ./aws_edge_benchmark_results
```

## Cleanup

To avoid ongoing EC2 cost:

```bash
terraform destroy
```

If you want the instance to stop itself after the benchmark:

```bash
terraform apply -var='shutdown_after_run=true'
```
