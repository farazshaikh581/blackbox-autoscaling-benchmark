#!/usr/bin/env bash
# Remove the calibration environment and free the cluster resources.
# Re-create it any time with deploy.sh.
set -euo pipefail
kubectl delete namespace blackbox --ignore-not-found
echo "namespace 'blackbox' removed."
