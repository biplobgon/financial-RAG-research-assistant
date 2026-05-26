output "cluster_name" {
  value       = google_container_cluster.financial_rag.name
  description = "GKE cluster name"
}

output "cluster_endpoint" {
  value       = google_container_cluster.financial_rag.endpoint
  description = "GKE cluster API endpoint"
  sensitive   = true
}

output "data_bucket_name" {
  value       = google_storage_bucket.financial_rag_data.name
  description = "GCS data storage bucket name"
}
