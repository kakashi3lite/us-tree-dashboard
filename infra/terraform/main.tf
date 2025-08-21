###############################################
# Terraform Root Module - Conservation System #
###############################################
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
  backend "s3" {
    bucket         = var.state_bucket
    key            = "tfstate/conservation-dashboard/primary.tfstate"
    region         = var.region
    dynamodb_table = var.lock_table
    encrypt        = true
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project     = "ConservationDashboard"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.owner
    }
  }
}

############################
# Networking (VPC + Subnets)
############################
module "network" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "conservation-${var.environment}"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  public_subnets  = var.public_subnets
  private_subnets = var.private_subnets

  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_nat_gateway = true
  single_nat_gateway = true
  enable_flow_log    = true
  flow_log_destination_type = "s3"
  flow_log_destination_arn  = aws_s3_bucket.logs.arn
}

############################
# ECS Cluster (Fargate)
############################
resource "aws_ecs_cluster" "this" {
  name = "conservation-${var.environment}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  configuration {
    execute_command_configuration {
      logging = "DEFAULT"
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "fargate" {
  cluster_name = aws_ecs_cluster.this.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

############################
# S3 Buckets
############################
resource "aws_s3_bucket" "data" {
  bucket = "${var.bucket_prefix}-data-${var.environment}"
  force_destroy = false
}

resource "aws_s3_bucket" "logs" {
  bucket = "${var.bucket_prefix}-logs-${var.environment}"
  force_destroy = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

############################
# CloudWatch Log Group
############################
resource "aws_cloudwatch_log_group" "app" {
  name              = "/conservation/${var.environment}/app"
  retention_in_days = 30
}

############################
# ALB
############################
resource "aws_lb" "public" {
  name               = "conservation-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  subnets            = module.network.public_subnets
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "app" {
  name        = "conservation-app-${var.environment}"
  port        = 8050
  protocol    = "HTTP"
  vpc_id      = module.network.vpc_id
  target_type = "ip"
  health_check {
    path                = "/healthz"
    matcher             = "200"
    healthy_threshold   = 3
    unhealthy_threshold = 2
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.public.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

############################
# Security Groups
############################
resource "aws_security_group" "alb" {
  name        = "conservation-alb-${var.environment}"
  description = "ALB ingress"
  vpc_id      = module.network.vpc_id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "conservation-svc-${var.environment}"
  description = "Service tasks"
  vpc_id      = module.network.vpc_id
  ingress {
    from_port       = 8050
    to_port         = 8050
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

############################
# Task Execution IAM Role (simplified)
############################
data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    effect = "Allow"
    principals { type = "Service" identifiers = ["ecs-tasks.amazonaws.com"] }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "conservation-execution-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

############################
# Variables & Data
############################
data "aws_availability_zones" "available" {}
