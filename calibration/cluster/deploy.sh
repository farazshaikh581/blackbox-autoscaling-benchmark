#!/usr/bin/env bash
# Stand up the isolated calibration environment in namespace `blackbox`.
# Uses the public factorizator image (Python 3.9 + Flask) with our compute-bound
# app.py and loadgen.py mounted via ConfigMaps, so nothing is built or pushed.
# A ResourceQuota caps the namespace so the rest of the cluster keeps a core free.
set -euo pipefail
cd "$(dirname "$0")"
NS=blackbox
IMG=farazshaikh581/factorizator:v2

kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# Cap the namespace: at most 5 CPU / 8Gi of limits across all pods here.
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: ResourceQuota
metadata:
  name: blackbox-quota
  namespace: blackbox
spec:
  hard:
    limits.cpu: "5"
    limits.memory: 8Gi
YAML

# App + loadgen source as ConfigMaps (re-created so edits take effect).
kubectl -n "$NS" create configmap app-src --from-file=app.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" create configmap load-src --from-file=loadgen.py \
  --dry-run=client -o yaml | kubectl apply -f -

# Target deployment: our app.py mounted over /app/app.py; default CMD runs it.
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: target
  namespace: $NS
  labels: {app: calib-target}
spec:
  replicas: 1
  # Recreate so each config has exactly one pod; a failing config (e.g. too
  # little memory) leaves no old pod serving, so it is measured as infeasible.
  strategy: {type: Recreate}
  selector: {matchLabels: {app: calib-target}}
  template:
    metadata: {labels: {app: calib-target}}
    spec:
      # Keep off worker1, where the other studies run, so we never share a node
      # with their workloads (protects their baseline and our measurements).
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {key: kubernetes.io/hostname, operator: NotIn, values: ["k8s-worker1"]}
      containers:
      - name: target
        image: $IMG
        ports: [{containerPort: 8080}]
        env: [{name: WORK_UNITS, value: "40"}]
        resources:
          requests: {cpu: "100m", memory: "128Mi"}
          limits:   {cpu: "500m", memory: "256Mi"}
        readinessProbe:
          httpGet: {path: /health, port: 8080}
          initialDelaySeconds: 3
          periodSeconds: 5
        volumeMounts:
        - {name: app-src, mountPath: /app/app.py, subPath: app.py}
      volumes:
      - {name: app-src, configMap: {name: app-src}}
---
apiVersion: v1
kind: Service
metadata: {name: target, namespace: $NS}
spec:
  selector: {app: calib-target}
  ports: [{name: http, port: 8080, targetPort: 8080}]
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: loadgen, namespace: $NS, labels: {app: loadgen}}
spec:
  replicas: 1
  selector: {matchLabels: {app: loadgen}}
  template:
    metadata: {labels: {app: loadgen}}
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {key: kubernetes.io/hostname, operator: NotIn, values: ["k8s-worker1"]}
      containers:
      - name: loadgen
        image: $IMG
        command: ["sleep", "infinity"]
        resources:
          requests: {cpu: "100m", memory: "128Mi"}
          limits:   {cpu: "500m", memory: "256Mi"}
        volumeMounts:
        - {name: load-src, mountPath: /load}
      volumes:
      - {name: load-src, configMap: {name: load-src}}
YAML

echo "waiting for rollout..."
kubectl -n "$NS" rollout status deploy/target --timeout=120s
kubectl -n "$NS" rollout status deploy/loadgen --timeout=120s
echo "ready. namespace '$NS' up."
