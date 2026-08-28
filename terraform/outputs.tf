output "ec2_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.mlops_app.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.mlops_app.public_dns
}

output "health_endpoint" {
  description = "Health check URL"
  value       = "http://${aws_instance.mlops_app.public_ip}:8000/health"
}

output "metrics_endpoint" {
  description = "Prometheus metrics URL"
  value       = "http://${aws_instance.mlops_app.public_ip}:8000/metrics"
}