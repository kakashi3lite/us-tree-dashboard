variable "region" { type = string default = "us-east-1" }
variable "environment" { type = string default = "dev" }
variable "owner" { type = string default = "platform" }
variable "bucket_prefix" { type = string default = "conservation-dashboard" }
variable "state_bucket" { type = string }
variable "lock_table" { type = string }
variable "vpc_cidr" { type = string default = "10.40.0.0/16" }
variable "public_subnets" { type = list(string) default = ["10.40.0.0/24", "10.40.1.0/24", "10.40.2.0/24"] }
variable "private_subnets" { type = list(string) default = ["10.40.10.0/24", "10.40.11.0/24", "10.40.12.0/24"] }