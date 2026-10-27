#!/usr/bin/env bash
# Qwen3.8-Flash-Next (FP4-Local by default; hibrid48, hibrid48-uncensored, INT4-AutoRound) on ONE DGX Spark, vLLM 0.32: download if needed, wait for
# memory, evict stale page cache, serve, wait healthy.
# Everything is configured in recipe.yaml. OpenAI API on :$PORT. ./stop.sh stops, ./view.sh stats.
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

# --- tiny recipe.yaml reader (two-level: section -> key: value; strips quotes/comments) ---------
rkey() {  # rkey <section> <key>
  awk -v s="$1" -v k="$2" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec) }
    sec==s && $1==k":" {
      sub(/^[ ]*[^:]*:[ ]*/,""); sub(/[ ]+#.*$/,"")
      gsub(/^["\x27]|["\x27]$/,""); print; exit
    }' recipe.yaml
}
rsection() {  # all key/value lines of a section, "key<TAB>value" (quotes/comments stripped)
  awk -v s="$1" '
    /^[A-Za-z_]/ { sec=$1; sub(":$","",sec); next }
    sec==s && $1 ~ /^[A-Za-z0-9_-]+:$/ || (sec==s && /^[ ]+[A-Za-z0-9_-]+:[ ]/) {
      line=$0; sub(/^[ ]+/,"",line)
      key=line; sub(/:.*/,"",key)
      val=line; sub(/^[^:]*:[ ]*/,"",val); sub(/[ ]+#.*$/,"",val)
      gsub(/^["\x27]|["\x27]$/,"",val)
      if (key != "") print key "\t" val
    }' recipe.yaml
}

IMAGE="$(rkey server image)";       PORT="$(rkey server port)"
HF_REPO="$(rkey server model)"
MODELS_DIR="$(rkey server models_dir)"; CACHE_DIR="$(rkey server cache_dir)"
CPUSET="$(rkey server cpuset)"
DOCKER_ARGS="$(rkey server docker_args)"
NAME="qwen38-flash-next"
mkdir -p "$MODELS_DIR" "$CACHE_DIR"
MODELS_ABS="$(cd "$MODELS_DIR" && pwd)"; CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
LOCAL_NAME="$(basename "$HF_REPO")"
MODEL_DIR="$MODELS_ABS/$LOCAL_NAME"

# --- Hugging Face access ------------------------------------------------------------------------------
# Anonymous downloads are rate-limited, and GATED repos (license-agreement models — the uncensored variants, most
# fine-tunes of gated bases) refuse anonymous access outright: the download stalls, then dies with 401 after a while.
# Token order: $HF_TOKEN → ~/.cache/huggingface/token (from `hf auth login`) → ask, when interactive. For a gated
# repo the token must also have been GRANTED access (the agreement is per model) — checked before anything downloads.
# Nothing is stored by this script; `hf auth login` is how a token is kept.
hf_access() {  # hf_access <hf-repo>   — exports HF_TOKEN when one is found or entered; returns 1 when the download cannot work
  local repo="$1" meta gated code who
  if [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"; export HF_TOKEN
  fi
  meta="$(curl -s --max-time 15 "https://huggingface.co/api/models/$repo" || true)"
  case "$meta" in
    *'"gated":"auto"'*|*'"gated":"manual"'*|*'"gated":true'*) gated=yes ;;
    *'"gated":false'*) gated=no ;;
    "") echo "  (huggingface.co not reachable — skipping the access check)"; return 0 ;;
    *) echo "✗ $repo: not found on Hugging Face (or private)"; return 1 ;;
  esac
  if [ -z "${HF_TOKEN:-}" ]; then
    if [ "$gated" = yes ]; then
      echo "· $repo is a GATED model — Hugging Face only serves it to an account that accepted its agreement:"
      echo "    1. open https://huggingface.co/$repo and accept the agreement (some repos approve by hand — wait for the mail)"
      echo "    2. create a READ token at https://huggingface.co/settings/tokens"
      echo "    3. export HF_TOKEN=hf_...   or   hf auth login   (keeps it in ~/.cache/huggingface), then rerun"
      if [ -t 0 ]; then
        read -rsp "  paste the token now to continue (input hidden; Enter aborts): " HF_TOKEN; echo
        [ -n "$HF_TOKEN" ] || return 1
        export HF_TOKEN
      else
        return 1
      fi
    else
      echo "· no Hugging Face token (HF_TOKEN unset, no hf auth login) — anonymous downloads are rate-limited; a free READ token"
      echo "  from https://huggingface.co/settings/tokens is faster:  export HF_TOKEN=hf_...   or   hf auth login"
      if [ -t 0 ]; then
        read -rsp "  paste a token to use it now, or Enter to continue anonymously: " HF_TOKEN; echo
        if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; else unset HF_TOKEN; fi
      fi
      [ -n "${HF_TOKEN:-}" ] || return 0
    fi
  fi
  who="$(curl -s --max-time 15 -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 | sed -n 's/.*"name":"\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$who" ] || { echo "✗ the Hugging Face token is not valid (whoami failed) — check HF_TOKEN / hf auth login"; return 1; }
  code="$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 30 -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$repo/resolve/main/config.json")"
  case "$code" in
    200) if [ "$gated" = yes ]; then echo "· Hugging Face: $who — access to the gated $repo granted"; else echo "· Hugging Face: $who"; fi ;;
    401|403) echo "✗ $who has no access to $repo yet — accept the agreement at https://huggingface.co/$repo (manual approval takes a while), then rerun"; return 1 ;;
    *) echo "  (access check returned HTTP $code — continuing)" ;;
  esac
}
# --------------------------------------------------------------------------------------------------------

# --- weights: ~98G, resumable (rerun on interruption) -------------------------------------------
# "complete" = the index is there AND no partial blob is left behind by an interrupted download (huggingface_hub keeps
# them under .cache/huggingface/download/*.incomplete and resumes them) — the index lands early, so it alone proves nothing.
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || [ -n "$(find "$MODEL_DIR/.cache" -name '*.incomplete' -print -quit 2>/dev/null)" ]; then
  hf_access "$HF_REPO" || exit 1
  echo "· downloading $HF_REPO -> $MODEL_DIR"
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR" --exclude "fast-fp8/*"
  else
    # -t gives tqdm a TTY so per-file progress bars actually render; HF_TOKEN passes
    # through if exported (higher rate limits) and is harmless when unset.
    TTY=""; [ -t 1 ] && TTY="-t"
    # the container runs as root — hand the files back to the host user afterwards
    docker run --rm $TTY -e HF_TOKEN -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; import subprocess; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME', ignore_patterns=['fast-fp8/*']); subprocess.run(['chown', '-R', '$(id -u):$(id -g)', '/dl/$LOCAL_NAME'], check=False)"
  fi
fi

# --- assemble docker env + vllm flags straight from the recipe ----------------------------------
ENVS=()
while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)

# INT4-AutoRound checkpoint (fp8 n-gram table in ple-table/, its own MTP draft head): table folder + its own table map, and the
# draft folder the speculative-config points at (cache/draft-k10: links + small JSON, top-k 10, the head's shared-expert width)
if [ -d "$MODEL_DIR/ple-table" ]; then
  ENVS+=(-e "MBX_PLE_FP8_DIR=/models/$LOCAL_NAME/ple-table" -e "MBX_PLE_NVME_DIR=/cache/ple-nvme-fp8")
  python3 - "$MODEL_DIR" "/models/$LOCAL_NAME" "$CACHE_ABS/draft-k10" <<'PY' || { echo "✗ could not build the draft folder"; exit 1; }
import json, os, shutil, struct, sys
snap, csnap, out = sys.argv[1:4]
wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
keep = {k: v for k, v in wm.items() if k.startswith(("mtp.", "lm_head.")) or k.endswith("embed_tokens.weight")}
if os.path.lexists(out):
    shutil.rmtree(out)
os.makedirs(out)
for f in sorted(os.listdir(snap)):
    if f in ("config.json", "model.safetensors.index.json", "ple-table", "fast-fp8", ".cache") or \
       (f.endswith(".safetensors") and f not in set(keep.values())):
        continue
    os.symlink(os.path.join(csnap, f), os.path.join(out, f))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
cfg = json.load(open(os.path.join(snap, "config.json"))); t = cfg.get("text_config", cfg)
t["num_experts_per_tok"] = 10
with open(os.path.join(snap, "model_extra_tensors.safetensors"), "rb") as fh:
    h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
w = {v["shape"][0] for k, v in h.items() if k.endswith("mlp.shared_expert.gate_proj.weight")}
if len(w) == 1:
    t["shared_expert_intermediate_size"] = w.pop()
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
PY
  echo "· INT4-AutoRound: fp8 table from ple-table/ (its map: cache/ple-nvme-fp8, ~52G, built on the first boot), draft head in cache/draft-k10"
fi
# two-stage draft head, opt-in (MBX_DRAFT_HEAD2: auto): the basis fitted on this checkpoint, shipped in the image
for i in "${!ENVS[@]}"; do
  if [ "${ENVS[$i]}" = "MBX_DRAFT_HEAD2=auto" ]; then
    BASIS="/opt/mbx/draft/$LOCAL_NAME.basis.pt"
    if docker run --rm --entrypoint test "$IMAGE" -f "$BASIS"; then ENVS[$i]="MBX_DRAFT_HEAD2=$BASIS"
    else ENVS[$i]="MBX_DRAFT_HEAD2="; echo "· no draft-head basis for $LOCAL_NAME in the image: the full head drafts"; fi
  fi
done
# compressed-tensors checkpoints (hibrid48-uncensored: fp8 q/k/v, bf16 indexer projection) load the indexer unfused
if grep -q '"quant_method": *"compressed-tensors"' "$MODEL_DIR/config.json" 2>/dev/null; then
  ENVS+=(-e MBX_QSA_UNFUSE_INDEXER=1)
fi
FLAGS=()
while IFS=$'\t' read -r k v; do
  case "$v" in
    true)        FLAGS+=("--$k");;
    false|null|"") ;;
    *)           FLAGS+=("--$k" "$v");;
  esac
done < <(rsection vllm)

# INT4-AutoRound profile (measured on this checkpoint): fp8 side layers + draft scale 2, its draft head, no global MoE backend
# (int4 experts take Marlin anyway; the draft head's bf16 experts refuse it), KV 24G (27G leaves too little headroom)
if [ -d "$MODEL_DIR/ple-table" ]; then
  ENVS+=(-e VLLM_FP8_HYBRID=1 -e MBX_MTP_DRAFT_SCALE=2)
  F2=(); i=0
  while [ $i -lt ${#FLAGS[@]} ]; do
    f="${FLAGS[$i]}"
    case "$f" in
      --moe-backend) i=$((i + 2)); continue;;
      --speculative-config) F2+=("$f" "${FLAGS[$((i + 1))]/\{\"method\":\"mtp\",/{\"method\":\"mtp\",\"model\":\"/cache/draft-k10\",}"); i=$((i + 2)); continue;;
      --kv-cache-memory) v="${FLAGS[$((i + 1))]}"; [ "$v" = 27000000000 ] && v=24000000000; F2+=("$f" "$v"); i=$((i + 2)); continue;;
    esac
    F2+=("$f"); i=$((i + 1))
  done
  FLAGS=("${F2[@]}")
fi

# optional front proxy (recipe.yaml `proxy:` section; absent = none, vLLM serves :$PORT itself): keepalive pings during long
# prefills + a loop guard on the reasoning stream (logs to cache/logs/, stop: true also cuts the stream and aborts the seat)
PROXY="$(rsection proxy | head -1)"
if [ -n "$PROXY" ]; then
  VPORT="$(rkey proxy upstream_port)"; VPORT="${VPORT:-$((PORT + 1))}"
  PENV=(-e "MBX_PROXY_PORT=$PORT" -e "MBX_PROXY_UPSTREAM=http://127.0.0.1:$VPORT" -e "MBX_PROXY_LOGS=/cache/logs")
  for k in keepalive loop_guard size loop_period loop_repeats loop_fields pattern_guard pattern_count log_all stop token; do
    v="$(rkey proxy "$k")"; [ -n "$v" ] && PENV+=(-e "MBX_PROXY_$(echo "$k" | tr a-z A-Z)=$v")
  done
  PF="$(rkey proxy patterns_file)"; [ -n "$PF" ] && PENV+=(-e "MBX_PROXY_PATTERNS_FILE=/cache/$PF")
  PUBLISH=(-p "127.0.0.1:$VPORT:8000")
else
  PUBLISH=(-p "$PORT:8000")
fi

# kernel page compaction — read-only check (the fix needs root → ./tune-host.sh). On a Spark the GPU's memory is
# ordinary pages; the kernel's proactive compactor migrating them measured as 4-5 s stalls every ~37 s (~10 %).
cp_now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
if [ "$cp_now" != 0 ]; then
  echo "  ⚠ vm.compaction_proactiveness is $cp_now (want 0): expect ~10 % lower throughput and periodic 4-5 s stalls"
  echo "    under load. One-time fix, needs sudo, shows what it runs first:  ./tune-host.sh"
fi
docker rm -f "$NAME" "$NAME-proxy" >/dev/null 2>&1 || true

# optional vLLM patches (recipe.yaml server.patches → patches/<name>.patch; default none): the files a patch touches are
# copied out of the image, patched and mounted read-only over the image's own copies. The image itself is never changed.
PATCHES="$(rkey server patches)"; PMOUNTS=()
if [ -n "$PATCHES" ]; then
  command -v patch >/dev/null || { echo "✗ server.patches needs the 'patch' tool on this box (apt install patch)"; exit 1; }
  VLLM_DIR="$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import importlib.util as u; print(u.find_spec("vllm").submodule_search_locations[0])')"
  STAGE="$CACHE_ABS/patched"; rm -rf "$STAGE"; mkdir -p "$STAGE"
  cid="$(docker create "$IMAGE")"; FILES=()
  pfail() { docker rm "$cid" >/dev/null 2>&1; echo "✗ $*"; exit 1; }
  for p in ${PATCHES//,/ }; do
    pf="patches/$p.patch"; [ -f "$pf" ] || pfail "server.patches: $pf not found"
    while read -r rel; do
      [ -f "$STAGE/$rel" ] && continue
      mkdir -p "$STAGE/$(dirname "$rel")"
      docker cp "$cid:$VLLM_DIR/$rel" "$STAGE/$rel" >/dev/null 2>&1 || pfail "patch $p: vllm/$rel is not in $IMAGE"
      FILES+=("$rel")
    done < <(sed -n 's#^+++ b/\([^[:space:]]*\).*#\1#p' "$pf")
    patch --dry-run -s -p1 -d "$STAGE" < "$pf" >/dev/null 2>&1 || pfail "patch $p does not fit $IMAGE — remove it from server.patches"
    patch -s -p1 --no-backup-if-mismatch -d "$STAGE" < "$pf"
    echo "· patch $p applied"
  done
  docker rm "$cid" >/dev/null
  for rel in "${FILES[@]}"; do PMOUNTS+=(-v "$STAGE/$rel:$VLLM_DIR/$rel:ro"); done
fi

# memory gate (unified memory: a serve relaunched seconds after a teardown gets a PHANTOM "CUDA out of memory" — the
# previous container's GPU pages take 30-60 s to come back). The load needs ~100G available; the table then fills the rest.
t=0; while :; do
  avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "${avail:-0}" -ge 100 ] && { echo "  ✓ memory: ${avail}G available"; break; }
  [ "$t" -ge 120 ] && { echo "  ✗ only ${avail}G available after 120 s (need ~100G) — another container on the box? (docker ps)"; exit 1; }
  [ "$t" = 0 ] && echo "  · waiting for memory to come back (${avail}G available, need 100G)…"
  sleep 5; t=$((t+5))
done
# evict our own checkpoint files from the page cache (no root: POSIX_FADV_DONTNEED via dd) — the GPU driver wants pages
# that are FREE, and a 60-70G stale shard cache during the load has stalled it
find "$MODEL_DIR" -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null || true
echo "  · page cache: checkpoint files evicted — MemFree $(awk '/^MemFree/{printf "%d", $2/1048576}' /proc/meminfo)G"
[ "$(rkey env MBX_DEPTH_MODE)" = dynamic ] && echo "  · draft depth: dynamic, ceiling $(rkey env MBX_DEPTH_CEIL) (recipe.yaml env)"
echo "· starting $NAME  ($IMAGE)  on :$PORT — healthy in ~7 min; the very first boot on a box also prepares the table map and compiles kernels (~15 min, once)"
docker run -d --name "$NAME" --gpus all --ipc=host \
  ${CPUSET:+--cpuset-cpus "$CPUSET"} \
  ${DOCKER_ARGS} \
  "${PUBLISH[@]}" \
  -v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache" \
  -v "$(readlink -f "$MODEL_DIR"):/models/$LOCAL_NAME" \
  "${PMOUNTS[@]}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace \
  -e VLLM_CACHE_ROOT=/cache/vllm-cache \
  "${ENVS[@]}" \
  --entrypoint vllm "$IMAGE" serve "/models/$LOCAL_NAME" --port 8000 "${FLAGS[@]}" >/dev/null
if [ -n "$PROXY" ]; then
  docker run -d --name "$NAME-proxy" --network host -v "$CACHE_ABS:/cache" "${PENV[@]}" \
    --entrypoint python3 "$IMAGE" -c "import mbx_proxy; mbx_proxy.main()" >/dev/null
  echo "· proxy on :$PORT → vLLM on 127.0.0.1:$VPORT (keepalive + loop guard; logs → $CACHE_DIR/logs/)"
fi

echo "· streaming engine logs until healthy (Ctrl-C detaches; the container keeps booting)"
docker logs -f "$NAME" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 480); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    # warm-up + self-check: one short thinking-on request. The first real request then hits a warm engine, and the
    # answer (OAK, middle letter A) shows thinking and output both work.
    WPORT="$PORT"; [ -n "$PROXY" ] && WPORT="$VPORT"
    WMODEL="$(rkey vllm served-model-name)"; WMODEL="${WMODEL:-Qwen/Qwen3.8-Flash-Next}"
    python3 - "$WPORT" "$WMODEL" <<'PY' || echo "· warm-up skipped"
import json, re, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
q = "What is the word starting with O ending with K and that word has less then 4 letters. And what is his middle letter?"
body = {"model": model, "max_tokens": 8000, "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True}, "messages": [{"role": "user", "content": q}]}
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
m = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]
ans = " ".join((m.get("content") or "").split())
ok = bool(re.search(r"\b(OAK|OK)\b", ans.upper()))   # OAK (middle A) or OK (no middle letter)
print(f"{'✓' if ok else '⚠'} warm-up: {time.time() - t0:.1f} s · {ans[:90]}")
PY
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ server booted — OpenAI-compatible API is live"
    echo "    endpoint : http://127.0.0.1:$PORT/v1"
    echo "    monitor  : ./view.sh          (throughput, KV, acceptance)"
    echo "    logs     : docker logs -f $NAME"
    echo "    stop     : ./stop.sh"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo "✗ container exited — see above"; exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 40 min — still booting? watch: docker logs -f $NAME"; exit 1
