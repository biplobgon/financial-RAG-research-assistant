# ==============================================================================
# Terraform — Financial RAG Research Assistant Infrastructure
# GCP + GKE deployment
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.29"
    }
  }

  backend "gcs" {
    bucket = "financial-rag-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─────────────────────────────────────────────────
# GKE Cluster
# ─────────────────────────────────────────────────
resource "google_container_cluster" "financial_rag" {
  name     = "financial-rag-cluster"
  location = var.region
  project  = var.project_id

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.financial_rag_vpc.name
  subnetwork = google_compute_subnetwork.financial_rag_subnet.name

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

resource "google_container_node_pool" "api_nodes" {
  name     = "api-node-pool"
  cluster  = google_container_cluster.financial_rag.name
  location = var.region
  project  = var.project_id

  initial_node_count = 2

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }

  node_config {
    machine_type = "n2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      env     = var.environment
      project = "financial-rag"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ─────────────────────────────────────────────────
# VPC Network
# ─────────────────────────────────────────────────
resource "google_compute_network" "financial_rag_vpc" {
  name                    = "financial-rag-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

resource "google_compute_subnetwork" "financial_rag_subnet" {
  name          = "financial-rag-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.financial_rag_vpc.id
  project       = var.project_id
}

# ─────────────────────────────────────────────────
# GCS Bucket (Data Storage)
# ─────────────────────────────────────────────────
resource "google_storage_bucket" "financial_rag_data" {
  name          = "${var.project_id}-financial-rag-data"
  location      = var.region
  project       = var.project_id
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Archive"
    }
  }
}
