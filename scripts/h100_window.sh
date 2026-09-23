#!/usr/bin/env bash
# H100 window driver. Runs ON the the GPU cluster GPU node as the admin user, from ~/laya-bench-prep (this script, the k8s/
# manifests, laya-bench.tgz and harness-overlay.tgz live there). One subcommand per row of scripts/h100_window.md;
# every prod-touching one (hold, colleague-restart, mig, restore) is printed to and confirmed by the user first.
#
#   status                         what owns the GPUs, MIG labels, hold policy, controller replicas
#   snapshot                       node / cluster-policy / custom-mig-config / GpuMigConfig / namespace objects
#   migconfig-add                  append mixed-bd + all-1g-bd to custom-mig-config (inert until the node label changes)
#   hold                           Kyverno hold, delete our predictor pod, wait for GPU 0 idle, pin clocks
#   pod <name> <resource> [slice]  create a bench pod (resource = nvidia.com/gpu | nvidia.com/mig-2g.24gb | nvidia.com/mig-1g.12gb),
#                                  copy tarball + overlay (+ pip cache) in, run the setup stage
#   run <pod> <stage> [args...]    run a phase_d_h100_pod.sh stage inside the pod (logged under logs/)
#   pull <pod>                     copy /work/results/h100nvl out into results/h100nvl (merge)
#   pipcache-save <pod>            copy the pod's pip cache out so the slice pods install without downloading
#   qwen4b-up                      temporary Qwen3.5-4B vLLM pod on the whole GPU (then port-forward 8010:8080)
#   colleague-restart              delete the colleague's qwen3-8b predictor pod (agreed; needed for a MIG transition)
#   mig <config>                   strategy mixed, controller to 0, label the node, wait for state=success
#   slices7 up|solo|sweep|replay|pull|down   the 7 x 1g.12gb aggregate experiment
#   restore                        back to whole / single / original config / controller 1 / clocks / no hold; wait for the predictor
set -u
export KUBECONFIG=/etc/kubernetes/admin.conf
K="sudo --preserve-env=KUBECONFIG kubectl"
NS=<your-namespace>
COLLEAGUE_NS=<other-tenant-namespace>
NODE=<gpu-node>
PREP=$HOME/laya-bench-prep
BOX=h100nvl
cd "$PREP" || exit 1
mkdir -p logs results/$BOX
smi() { sudo chroot /run/nvidia/driver nvidia-smi "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
gpu0_used() { smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
mig_state() { $K get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config}{" "}{.metadata.labels.nvidia\.com/mig\.config\.state}'; }
wait_mig_success() {  # $1 = expected label
  for i in $(seq 1 90); do
    s=$(mig_state); [ "$s" = "$1 success" ] && { log "MIG $s"; return 0; }
    [ $((i % 6)) -eq 0 ] && log "MIG label/state: $s"
    sleep 10
  done
  log "MIG transition did not reach success in 15 min; mig-manager log tail:"
  $K -n gpu-operator logs ds/nvidia-mig-manager --tail=30
  return 1
}
wait_gpu0_idle() {
  for i in $(seq 1 60); do
    u=$(gpu0_used); [ "$u" -lt 1024 ] && { log "GPU 0 idle (${u} MiB)"; return 0; }
    [ $((i % 6)) -eq 0 ] && { log "GPU 0 still ${u} MiB used"; smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; }
    sleep 10
  done
  log "GPU 0 not idle after 10 min"; return 1
}

status() {
  echo "--- our namespace"; $K -n $NS get pods -o wide | grep -vE "^fs-|ml-pipeline"
  echo "--- colleague"; $K -n $COLLEAGUE_NS get pods | grep -E "NAME|predictor"
  echo "--- node MIG: $(mig_state)  strategy(label)=$($K get node $NODE -o jsonpath='{.metadata.labels.nvidia\.com/mig\.strategy}')  strategy(policy)=$($K get clusterpolicies.nvidia.com cluster-policy -o jsonpath='{.spec.mig.strategy}')"
  echo "--- hold policy: $($K get clusterpolicies.kyverno.io laya-bench-hold-gpu -o name 2>/dev/null || echo none)   gpunodeconfig controller replicas: $($K -n gpu-nodeconfig get deploy gpunodeconfig-controller-manager -o jsonpath='{.spec.replicas}')"
  echo "--- MIG resources: $($K get node $NODE -o jsonpath='{.status.allocatable}' | tr ',' '\n' | grep -E 'nvidia.com' | tr '\n' ' ')"
  echo "--- gpus"; smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,mig.mode.current --format=csv
  smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
}

snapshot() {
  d=snapshot-$(date -u +%Y%m%dT%H%M); mkdir -p "$d"
  $K get node $NODE -o yaml > $d/node.yaml
  $K get clusterpolicies.nvidia.com cluster-policy -o yaml > $d/clusterpolicy.yaml
  $K -n gpu-operator get cm custom-mig-config -o yaml > $d/custom-mig-config.yaml
  $K -n gpu-operator get cm custom-mig-config -o jsonpath='{.data.config\.yaml}' > $d/custom-mig-config.config.yaml
  $K get gpumigconfigs -A -o yaml > $d/gpumigconfigs.yaml
  $K -n $NS get resourcequota,isvc,revisions,pods -o yaml > $d/namespace.yaml
  $K -n gpu-nodeconfig get deploy gpunodeconfig-controller-manager -o yaml > $d/gpunodeconfig-controller.yaml
  ln -sfn "$d" snapshot-latest; log "snapshot in $PREP/$d (snapshot-latest -> it)"; ls -la $d
}

cm_patch() {  # $1 = file holding the new config.yaml
  python3 -c 'import json,sys; print(json.dumps({"data":{"config.yaml":open(sys.argv[1]).read()}}))' "$1" \
    | $K -n gpu-operator patch cm custom-mig-config --type merge -p "$(cat)"
}
migconfig_add() {
  [ -e snapshot-latest ] || { echo "run snapshot first"; exit 1; }
  cur=snapshot-latest/custom-mig-config.config.yaml
  grep -q "^  mixed-bd:" $cur && { log "mixed-bd already present"; return 0; }
  { cat $cur; echo; sed -n '/^mig-configs:/,$p' k8s/mig-configs.yaml | tail -n +2; } > /tmp/custom-mig-config.new.yaml
  cm_patch /tmp/custom-mig-config.new.yaml
  $K -n gpu-operator get cm custom-mig-config -o jsonpath='{.data.config\.yaml}' | grep -nE "^  [a-z-]+:"
}

hold() {
  $K apply -f k8s/kyverno-hold.yaml
  $K -n $NS delete pod -l serving.kserve.io/inferenceservice=qwen3-5-35b-a3b-fp8 --wait=false
  wait_gpu0_idle || exit 1
  smi -i 0 -lgc 1785,1785; smi -i 0 --query-gpu=clocks.sm,clocks.max.sm --format=csv
}

make_pod() {  # $1 name, $2 resource, $3 "slice" for small limits (7 pods must fit the namespace quota: 40 CPU / 100 GiB limits)
  sed -e "s/^  name: laya-bench$/  name: $1/" -e "s#nvidia.com/gpu: \"1\".*#$2: \"1\"#" k8s/bench-pod.yaml |
    if [ "${3:-}" = slice ]; then sed -e 's/cpu: "8", memory: 32Gi/cpu: "2", memory: 8Gi/' -e 's/cpu: "16"/cpu: "5"/' -e 's/memory: 64Gi/memory: 13Gi/'; else cat; fi
}
pod() {
  name=$1; res=$2; kind=${3:-}
  make_pod "$name" "$res" "$kind" | $K apply -f - || exit 1
  $K -n $NS wait --for=condition=Ready pod/$name --timeout=600s || { $K -n $NS describe pod $name | tail -15; exit 1; }
  log "$name ready; copying tarball + overlay"
  $K -n $NS cp laya-bench.tgz $name:/work/laya-bench.tgz
  $K -n $NS cp harness-overlay.tgz $name:/work/harness-overlay.tgz
  [ -d pipcache ] && $K -n $NS cp pipcache $name:/work/pipcache
  $K -n $NS exec $name -- bash -c 'cd /work && tar xzf laya-bench.tgz && tar xzf harness-overlay.tgz && rm laya-bench.tgz harness-overlay.tgz && ls'
  run "$name" setup
}
run() {
  p=$1; shift
  log "run $p: $*"
  $K -n $NS exec $p -- bash -c "cd /work && bash scripts/phase_d_h100_pod.sh $*" 2>&1 | tee -a "logs/$p.$1.log"
}
pull() {
  rm -rf /tmp/pull-$1; mkdir -p /tmp/pull-$1
  $K -n $NS cp $1:/work/results/$BOX /tmp/pull-$1 && cp -a /tmp/pull-$1/. results/$BOX/ && rm -rf /tmp/pull-$1
  find results/$BOX -newer logs -type f | head -40
}
pipcache_save() { rm -rf pipcache; $K -n $NS cp $1:/work/pipcache pipcache && du -sh pipcache; }

qwen4b_up() {
  $K apply -f k8s/vllm-4b-pod.yaml
  $K -n $NS wait --for=condition=Ready pod/laya-bench-qwen4b --timeout=1800s || { $K -n $NS logs laya-bench-qwen4b --tail=20; exit 1; }
  log "Qwen3.5-4B ready; now: $K -n $NS port-forward pod/laya-bench-qwen4b 8010:8080"
}

colleague_restart() {
  echo "colleague's predictor pod (agreed restart for the MIG transition):"; $K -n $COLLEAGUE_NS get pods -l serving.kserve.io/inferenceservice=qwen3-8b
  $K -n $COLLEAGUE_NS delete pod -l serving.kserve.io/inferenceservice=qwen3-8b --wait=false
}
mig() {
  cfg=$1
  [ -z "$($K -n $NS get pods -l app=laya-bench -o name)" ] || { echo "bench pods still exist; delete them first"; exit 1; }
  echo "--- GPU processes (must be none for a transition)"; smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
  $K patch clusterpolicies.nvidia.com cluster-policy --type merge -p '{"spec":{"mig":{"strategy":"mixed"}}}'
  $K -n gpu-nodeconfig scale deploy gpunodeconfig-controller-manager --replicas=0
  $K label node $NODE nvidia.com/mig.config=$cfg --overwrite
  wait_mig_success "$cfg" || exit 1
  smi -L; sleep 20; status | grep "MIG resources"
}

slices7() {
  case "$1" in
  up)     for i in 0 1 2 3 4 5 6; do pod laya-bench-$i nvidia.com/mig-1g.12gb slice > logs/laya-bench-$i.up.log 2>&1 & done; wait
          for i in 0 1 2 3 4 5 6; do echo "== laya-bench-$i: $(tail -3 logs/laya-bench-$i.up.log | tr '\n' ' ')"; done ;;
  solo)   run laya-bench-0 slice mig-1g.12gb-solo ;;                           # one slice alone = the single-slice number
  sweep)  for i in 0 1 2 3 4 5 6; do run laya-bench-$i sweep-only mig-1g.12gb-slice$i > /dev/null 2>&1 & done; wait
          grep -h "SLO" logs/laya-bench-*.sweep-only.log ;;
  replay) for i in 0 1 2 3 4 5 6; do run laya-bench-$i replay mig-1g.12gb-slice$i 1430000 48 > /dev/null 2>&1 & done; wait
          tail -n 4 logs/laya-bench-*.replay.log ;;
  pull)   for i in 0 1 2 3 4 5 6; do pull laya-bench-$i; done ;;
  down)   $K -n $NS delete pod -l app=laya-bench --wait=true ;;
  *) echo "slices7 up|solo|sweep|replay|pull|down"; exit 2;;
  esac
}

restore() {
  $K -n $NS delete pod -l app=laya-bench --wait=true 2>/dev/null
  wait_gpu0_idle || echo "WARNING: GPU 0 not idle, MIG relabel may fail with ERROR_IN_USE"
  if [ "$(mig_state)" != "whole success" ]; then
    $K label node $NODE nvidia.com/mig.config=whole --overwrite
    wait_mig_success whole || exit 1
  fi
  $K patch clusterpolicies.nvidia.com cluster-policy --type merge -p '{"spec":{"mig":{"strategy":"single"}}}'
  [ -e snapshot-latest ] && cm_patch snapshot-latest/custom-mig-config.config.yaml
  $K -n gpu-nodeconfig scale deploy gpunodeconfig-controller-manager --replicas=1
  smi -i 0 -rgc
  $K delete clusterpolicies.kyverno.io laya-bench-hold-gpu --ignore-not-found
  log "waiting for our predictor pod (weights re-download, ~25 min) ..."
  for i in $(seq 1 240); do
    r=$($K -n $NS get pods -l serving.kserve.io/inferenceservice=qwen3-5-35b-a3b-fp8 -o jsonpath='{.items[*].status.containerStatuses[*].ready}' 2>/dev/null)
    case "$r" in *false*|"") [ $((i % 12)) -eq 0 ] && log "predictor not ready yet: $r";; *) log "predictor ready: $r"; break;; esac
    sleep 10
  done
  status
  echo "NEXT (on workstation): VIP smoke, then <app-services>/scripts/<llm-failover-script> cluster; then verify the colleague's qwen3-8b is Ready."
}

case "${1:-status}" in
  status) status;; snapshot) snapshot;; migconfig-add) migconfig_add;; hold) hold;;
  pod) pod "$2" "$3" "${4:-}";; run) shift; run "$@";; pull) pull "$2";; pipcache-save) pipcache_save "$2";;
  qwen4b-up) qwen4b_up;; colleague-restart) colleague_restart;; mig) mig "$2";; slices7) slices7 "$2";; restore) restore;;
  *) sed -n '2,22p' "$0"; exit 2;;
esac
