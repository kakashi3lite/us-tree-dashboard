output "vpc_id" { value = module.network.vpc_id }
output "public_subnets" { value = module.network.public_subnets }
output "alb_dns_name" { value = aws_lb.public.dns_name }