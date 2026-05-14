output "instance_id" {
  description = "EC2 instance ID running the benchmark."
  value       = aws_instance.runner.id
}

output "instance_public_ip" {
  description = "Public IP address, if assigned."
  value       = aws_instance.runner.public_ip
}

output "ssh_command" {
  description = "SSH command if public_key_path and allowed_ssh_cidrs were configured."
  value       = local.create_key_pair ? "ssh ubuntu@${aws_instance.runner.public_ip}" : "SSH key pair not created; use AWS Systems Manager Session Manager."
}

output "ssm_command" {
  description = "AWS CLI command to open a Session Manager shell."
  value       = "aws ssm start-session --target ${aws_instance.runner.id} --region ${var.aws_region}${var.aws_profile != "" ? " --profile ${var.aws_profile}" : ""}"
}

output "results_bucket" {
  description = "S3 bucket for benchmark outputs, if created."
  value       = local.results_bucket
}

output "results_s3_prefix" {
  description = "S3 prefix where benchmark outputs are uploaded."
  value       = var.create_results_bucket ? "s3://${local.results_bucket}/${var.results_s3_prefix}/" : "S3 upload disabled"
}

output "ec2_log_paths" {
  description = "Useful paths on the EC2 instance."
  value = {
    user_data_log = "/var/log/face-edge-sim/user_data.log"
    repo_dir      = "/opt/face-edge-sim/repo"
    outputs_dir   = "/opt/face-edge-sim/repo/outputs/aws_edge_benchmark"
  }
}
