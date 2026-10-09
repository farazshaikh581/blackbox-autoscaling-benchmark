#!/usr/bin/env bash
# Stand up the multi-tier chain (frontend -> logic -> backend) in namespace
# `blackbox`, to validate the open-chain topology the simulator assumes:
# a request traverses every tier once and end-to-end latency is the
# sum of the per-tier sojourns. Same app.py as the single-tier target, driven
# into a chain by the DOWNSTREAM_URL env; mounted via ConfigMap, no image build.
#
# Reuses the single-tier discipline: capped by the namespace ResourceQuota and
# pinned off k8s-worker1, where other studies run, so we never share their node.
#
# Per-tier work/cpu/replicas/mem are overridable so a sweep can move the
# bottleneck or realize an optimized front config (calibration/measure_front_multinode.py):
#   FE_UNITS/LOGIC_UNITS/BE_UNITS       (CPU work per tier; the calibrated chain
#                                        constant -- NOT part of the search config)
#   FE_CPU/LOGIC_CPU/BE_CPU             (CPU limit per tier, cores       -> c_i)
#   FE_REPLICAS/LOGIC_REPLICAS/BE_REPLICAS (replica count per tier       -> n_i)
#   FE_MEM/LOGIC_MEM/BE_MEM             (memory limit per tier, k8s qty  -> m_i)
#
# QUOTA_CPU/QUOTA_MEM size the namespace ResourceQuota (default 5/8Gi). On a
# dedicated cluster, pass a bigger quota to use the real capacity.
#
# FE_NODE/LOGIC_NODE/BE_NODE pin a tier to an exact node (`spec.nodeName`,
# e.g. the full k8s node name) instead of leaving placement to the scheduler.
# Empty (default) = unpinned, unchanged behavior. Used to force a specific
# tier boundary to cross physical nodes (calibration/measure_wan_latency.py).
#
# REQUEST_EQUALS_LIMIT=1 sets each pod's cpu/mem *request* equal to its limit
# instead of the fixed 100m/64Mi stub. The scheduler places pods by request, not
# limit, so with the stub every pod's real footprint is invisible to it and a
# config's replicas can all land on one node regardless of their combined CPU
# limit -- fine for a single-node CPU-throttle-knee slice, but it silently
# defeats a genuine multi-node deployment (calibration/measure_front_multinode.py),
# where pods must actually spread across nodes by their real cpu/mem footprint,
# matching what `energy_model.pack_pods` assumes when scoring the config.
set -euo pipefail
cd "$(dirname "$0")"
NS=blackbox
IMG=farazshaikh581/factorizator:v2

FE_UNITS="${FE_UNITS:-15}"; LOGIC_UNITS="${LOGIC_UNITS:-25}"; BE_UNITS="${BE_UNITS:-40}"
FE_CPU="${FE_CPU:-0.5}";    LOGIC_CPU="${LOGIC_CPU:-0.5}";    BE_CPU="${BE_CPU:-0.5}"
FE_REPLICAS="${FE_REPLICAS:-1}"; LOGIC_REPLICAS="${LOGIC_REPLICAS:-1}"; BE_REPLICAS="${BE_REPLICAS:-1}"
FE_MEM="${FE_MEM:-256Mi}";  LOGIC_MEM="${LOGIC_MEM:-256Mi}";  BE_MEM="${BE_MEM:-256Mi}"
QUOTA_CPU="${QUOTA_CPU:-5}"; QUOTA_MEM="${QUOTA_MEM:-8Gi}"
REQUEST_EQUALS_LIMIT="${REQUEST_EQUALS_LIMIT:-0}"
FE_NODE="${FE_NODE:-}"; LOGIC_NODE="${LOGIC_NODE:-}"; BE_NODE="${BE_NODE:-}"

kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f - <<YAML
apiVersion: v1
kind: ResourceQuota
metadata: {name: blackbox-quota, namespace: blackbox}
spec:
  hard: {limits.cpu: "$QUOTA_CPU", limits.memory: $QUOTA_MEM}
YAML

# App source as a ConfigMap (re-created so edits take effect).
kubectl -n "$NS" create configmap app-src --from-file=app.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" create configmap load-src --from-file=loadgen.py \
  --dry-run=client -o yaml | kubectl apply -f -

# Emit one tier (Deployment + Service).
# Args: name units cpu downstream_url replicas mem node
tier() {
  local name="$1" units="$2" cpu="$3" downstream="$4" replicas="$5" mem="$6" node="${7:-}"
  local req_cpu="100m" req_mem="64Mi"
  if [ "$REQUEST_EQUALS_LIMIT" = "1" ]; then req_cpu="$cpu"; req_mem="$mem"; fi
  local node_line=""
  if [ -n "$node" ]; then node_line="nodeName: $node"; fi
  kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: {name: $name, namespace: $NS, labels: {app: $name, chain: blackbox}}
spec:
  replicas: $replicas
  strategy: {type: Recreate}
  selector: {matchLabels: {app: $name}}
  template:
    metadata: {labels: {app: $name, chain: blackbox}}
    spec:
      $node_line
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {key: kubernetes.io/hostname, operator: NotIn, values: ["k8s-worker1"]}
      containers:
      - name: app
        image: $IMG
        ports: [{containerPort: 8080}]
        env:
        - {name: TIER, value: "$name"}
        - {name: WORK_UNITS, value: "$units"}
        - {name: DOWNSTREAM_URL, value: "$downstream"}
        resources:
          requests: {cpu: "$req_cpu", memory: "$req_mem"}
          limits:   {cpu: "$cpu", memory: "$mem"}
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
metadata: {name: $name, namespace: $NS}
spec:
  selector: {app: $name}
  ports: [{name: http, port: 8080, targetPort: 8080}]
YAML
}

# backend is the leaf (no downstream); logic -> backend; frontend -> logic.
tier backend  "$BE_UNITS"    "$BE_CPU"    ""                             "$BE_REPLICAS"    "$BE_MEM"    "$BE_NODE"
tier logic    "$LOGIC_UNITS" "$LOGIC_CPU" "http://backend.$NS.svc:8080"  "$LOGIC_REPLICAS" "$LOGIC_MEM" "$LOGIC_NODE"
tier frontend "$FE_UNITS"    "$FE_CPU"    "http://logic.$NS.svc:8080"    "$FE_REPLICAS"    "$FE_MEM"    "$FE_NODE"

# Load generator (idle; driven via kubectl exec, as in the single-tier sweep).
kubectl apply -f - <<YAML
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
for d in backend logic frontend loadgen; do
  kubectl -n "$NS" rollout status "deploy/$d" --timeout=120s
done
echo "ready. chain frontend -> logic -> backend up in namespace '$NS'."
echo "units fe/logic/be = $FE_UNITS/$LOGIC_UNITS/$BE_UNITS  cpu = $FE_CPU/$LOGIC_CPU/$BE_CPU"
echo "replicas fe/logic/be = $FE_REPLICAS/$LOGIC_REPLICAS/$BE_REPLICAS  mem = $FE_MEM/$LOGIC_MEM/$BE_MEM"
