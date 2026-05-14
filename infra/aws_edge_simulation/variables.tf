variable "aws_region" {
  description = "AWS region for the edge-constrained simulation."
  type        = string
  default     = "ap-southeast-1"
}

variable "aws_profile" {
  description = "Optional AWS CLI profile, for example an IAM Identity Center/SSO profile."
  type        = string
  default     = ""
}

variable "project_name" {
  description = "Name prefix used for AWS resources."
  type        = string
  default     = "face-edge-sim"
}

variable "repo_url" {
  description = "Git repository URL to clone on the EC2 instance."
  type        = string
  default     = "https://github.com/tuananh-hoang/research_face_recognition_edge_device.git"
}

variable "repo_branch" {
  description = "Git branch to check out before running the benchmark."
  type        = string
  default     = "conditional-pipeline"
}

variable "instance_type" {
  description = "ARM64 EC2 instance type for the simulation host."
  type        = string
  default     = "t4g.small"
}

variable "ami_id" {
  description = "Optional Ubuntu ARM64 AMI ID. Leave empty to use the latest Canonical Ubuntu 24.04 ARM64 AMI found by filters."
  type        = string
  default     = ""
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size. Docker image builds need more than the tiny default."
  type        = number
  default     = 30
}

variable "public_key_path" {
  description = "Optional local SSH public key path. If empty, no EC2 key pair is created."
  type        = string
  default     = ""
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH to the instance. Keep empty to avoid opening port 22 and use SSM instead."
  type        = list(string)
  default     = []
}

variable "benchmark_dataset" {
  description = "Dataset argument passed to run_conditional_experiment.py."
  type        = string
  default     = "synthetic"

  validation {
    condition     = contains(["synthetic", "custom"], var.benchmark_dataset)
    error_message = "benchmark_dataset must be synthetic or custom."
  }
}

variable "data_dir" {
  description = "Dataset directory inside the cloned repository."
  type        = string
  default     = "data"
}

variable "custom_data_s3_uri" {
  description = "Optional S3 URI to sync custom data into the repo before running, e.g. s3://bucket/path/data."
  type        = string
  default     = ""
}

variable "max_pairs" {
  description = "MAX_PAIRS passed to run_aws_edge_benchmark.sh."
  type        = number
  default     = 1000
}

variable "methods" {
  description = "Comma-separated method IDs for the benchmark."
  type        = string
  default     = "M0,M1,M2,M3,M4"
}

variable "face_model_name" {
  description = "InsightFace model pack or alias. mobilefacenet resolves to buffalo_sc/MBF@WebFace600K."
  type        = string
  default     = "mobilefacenet"
}

variable "face_det_size" {
  description = "InsightFace detector input size passed to FACE_DET_SIZE."
  type        = string
  default     = "320,320"
}

variable "robust_enhancement" {
  description = "Robust path enhancement mode."
  type        = string
  default     = "clahe"
}

variable "create_results_bucket" {
  description = "Create an S3 bucket and upload benchmark outputs as a tar.gz archive."
  type        = bool
  default     = true
}

variable "results_bucket_name" {
  description = "Optional explicit S3 bucket name. If empty, Terraform uses project-name + account-id + region."
  type        = string
  default     = ""
}

variable "results_s3_prefix" {
  description = "S3 prefix for benchmark result archives."
  type        = string
  default     = "aws-edge-benchmark"
}

variable "shutdown_after_run" {
  description = "Stop the EC2 instance after user_data finishes the benchmark."
  type        = bool
  default     = false
}
