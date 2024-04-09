variable "environment" {
  type    = string
  default = "dev"
}

variable "timezone" {
  type    = string
  default = "America/New_York"
}

variable "service" {
  type    = string
  default = "ad_score"
}

variable "google_credentials" {
  type        = string
  sensitive   = true
  default     = ""
  description = "Google Cloud service account credentials"
}

