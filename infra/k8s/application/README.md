# K8s 배포 가이드

## 배포 방법
Docker image `upstage-gangwon-backend:lite`

```bash
# 모든 리소스 배포
kubectl apply -f .

# 상태 확인
kubectl get all -n gangwon
```
## secret 으로 api key 등록
```bash
  kubectl create secret generic app-secret \
  --from-literal=UPSTAGE_API_KEY={UPSTAGE_API_KEY} \
  -n gangwon
```
## 로그 확인
   ```bash
   # Check all resources
   kubectl get all -n gangwon

   # Check pod logs
   kubectl logs -f deployment/upstage-gangwon-backend -n gangwon
   kubectl logs -f deployment/chromadb -n gangwon
   ```

## 사전 설정

- **UPSTAGE_API_KEY**: Update the base64 encoded value in `02-configmap.yaml`
- **Domain**: Change `gangwon.local` in `05-ingress.yaml` to your actual domain
- **Resources**: Adjust CPU/memory requests and limits in deployment manifests

```bash
`kubectl delete namespace gangwon`
```