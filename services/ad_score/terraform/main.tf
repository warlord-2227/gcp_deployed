resource "google_storage_bucket" "bucket" {
  name                        = "mast-${var.service}-meta-${var.environment}" # Every bucket name must be globally unique
  location                    = "us-central1"
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
}


