# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (minikube, GKE, EKS, AKS, or local)
- kubectl configured
- Docker image published: `a01566204/ml-service:1.0.0`

## Quick Start

### 1. Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Or apply individually
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 2. Verify Deployment
```bash
# Check pods
kubectl get pods -l app=ml-service

# Check service
kubectl get svc ml-service

# Check HPA
kubectl get hpa ml-service-hpa

# View logs
kubectl logs -f deployment/ml-service
```

### 3. Test the API
```bash
# Get service endpoint
kubectl get svc ml-service

# Port forward for local testing
kubectl port-forward svc/ml-service 8000:80

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Class_X_Percentage": 85.5,
    "Class_XII_Percentage": 78.0,
    "Study_Hours": 5.0,
    "Gender": "Male",
    "Caste": "General",
    "Coaching": "Yes",
    "Medium": "English"
  }'
```

## Scaling

### Manual Scaling
```bash
# Scale to 5 replicas
kubectl scale deployment ml-service --replicas=5

# Check status
kubectl get pods -l app=ml-service
```

### Auto-Scaling (HPA)

The HorizontalPodAutoscaler automatically scales between 2-10 replicas based on:
- CPU utilization > 70%
- Memory utilization > 80%
```bash
# Watch auto-scaling in action
kubectl get hpa ml-service-hpa --watch

# Generate load to trigger scaling
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://ml-service/health; done"
```

## Monitoring
```bash
# View deployment status
kubectl describe deployment ml-service

# View pod logs
kubectl logs -f -l app=ml-service --all-containers=true

# View events
kubectl get events --sort-by='.metadata.creationTimestamp'

# Resource usage
kubectl top pods -l app=ml-service
kubectl top nodes
```

## Rolling Updates
```bash
# Update to new version
kubectl set image deployment/ml-service ml-service=a01566204/ml-service:1.1.0

# Check rollout status
kubectl rollout status deployment/ml-service

# Rollback if needed
kubectl rollout undo deployment/ml-service
```

## Cleanup
```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete individually
kubectl delete deployment ml-service
kubectl delete service ml-service
kubectl delete hpa ml-service-hpa
kubectl delete configmap ml-service-config
```

## Architecture
```
                     ┌──────────────────┐
                     │   Ingress/LB     │
                     │  (External IP)   │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │    Service       │
                     │  (LoadBalancer)  │
                     └────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  Pod 1  │          │  Pod 2  │          │  Pod 3  │
   │ ML API  │          │ ML API  │          │ ML API  │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                     ┌────────▼─────────┐
                     │       HPA        │
                     │  Auto-Scaler     │
                     │   (2-10 pods)    │
                     └──────────────────┘
```

## Production Considerations

1. **Secrets**: Store sensitive data in Kubernetes Secrets
2. **Persistent Storage**: Use PersistentVolumes for model storage
3. **Monitoring**: Integrate with Prometheus/Grafana
4. **Logging**: Use ELK or Loki for centralized logging
5. **Security**: Implement NetworkPolicies and PodSecurityPolicies
6. **CI/CD**: Automate deployments with GitOps (ArgoCD/Flux)

