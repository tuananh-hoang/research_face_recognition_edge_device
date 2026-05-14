data "aws_caller_identity" "current" {}

data "aws_ami" "ubuntu_arm64" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name = "name"
    values = [
      "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*",
      "ubuntu/images/hvm-ssd/ubuntu-noble-24.04-arm64-server-*",
    ]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  common_tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
    Purpose   = "aws-edge-constrained-simulation"
  }

  ami_id          = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu_arm64[0].id
  create_key_pair = var.public_key_path != ""
  results_bucket  = var.create_results_bucket ? aws_s3_bucket.results[0].bucket : ""
}

resource "aws_key_pair" "runner" {
  count      = local.create_key_pair ? 1 : 0
  key_name   = "${var.project_name}-runner"
  public_key = file(var.public_key_path)

  tags = local.common_tags
}

resource "aws_security_group" "runner" {
  name        = "${var.project_name}-sg"
  description = "Security group for AWS edge-constrained simulation runner"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = var.allowed_ssh_cidrs
    content {
      description = "SSH from configured CIDR"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    description = "Allow outbound internet access"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_s3_bucket" "results" {
  count  = var.create_results_bucket ? 1 : 0
  bucket = var.results_bucket_name != "" ? var.results_bucket_name : "${var.project_name}-${data.aws_caller_identity.current.account_id}-${var.aws_region}"

  tags = local.common_tags
}

resource "aws_s3_bucket_public_access_block" "results" {
  count                   = var.create_results_bucket ? 1 : 0
  bucket                  = aws_s3_bucket.results[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  count  = var.create_results_bucket ? 1 : 0
  bucket = aws_s3_bucket.results[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "results" {
  count  = var.create_results_bucket ? 1 : 0
  bucket = aws_s3_bucket.results[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_iam_role" "runner" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "results_s3" {
  count = var.create_results_bucket ? 1 : 0
  name  = "${var.project_name}-results-s3"
  role  = aws_iam_role.runner.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.results[0].arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload"
        ]
        Resource = "${aws_s3_bucket.results[0].arn}/${var.results_s3_prefix}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "custom_data_s3" {
  count = var.custom_data_s3_uri != "" ? 1 : 0
  name  = "${var.project_name}-custom-data-s3"
  role  = aws_iam_role.runner.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "runner" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.runner.name
}

resource "aws_instance" "runner" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.runner.id]
  iam_instance_profile        = aws_iam_instance_profile.runner.name
  key_name                    = local.create_key_pair ? aws_key_pair.runner[0].key_name : null
  associate_public_ip_address = true
  user_data_replace_on_change = true

  root_block_device {
    volume_size = var.root_volume_size_gb
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url           = var.repo_url
    repo_branch        = var.repo_branch
    benchmark_dataset  = var.benchmark_dataset
    data_dir           = var.data_dir
    custom_data_s3_uri = var.custom_data_s3_uri
    max_pairs          = var.max_pairs
    methods            = var.methods
    far_budgets        = var.far_budgets
    defer_margin       = var.defer_margin
    calibration_split  = var.calibration_split
    calibration_seed   = var.calibration_seed
    face_model_name    = var.face_model_name
    face_det_size      = var.face_det_size
    robust_enhancement = var.robust_enhancement
    results_bucket     = local.results_bucket
    results_s3_prefix  = var.results_s3_prefix
    shutdown_after_run = var.shutdown_after_run
  })

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-runner"
  })

  depends_on = [
    aws_iam_role_policy_attachment.ssm,
    aws_iam_role_policy.results_s3,
    aws_iam_role_policy.custom_data_s3,
  ]
}
