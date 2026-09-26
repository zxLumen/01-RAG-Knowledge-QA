#!/usr/bin/env bash
# 内存巡检：记录宿主机与各容器的内存构成，重点区分「真实占用(anon)」与
# 「可回收的页面缓存(file)」。面板上把两者相加常得到接近整机容量的数字，
# 看不出到底是谁在吃内存；本脚本把 anon / file 拆开。
#
# 用法:
#   ./mem-watch.sh            # 追加一条快照到日志（供 cron 使用）
#   ./mem-watch.sh --stdout   # 只打印，不写日志
#
# 退出码: 可用内存 < WARN_AVAILABLE_MB 或 swap 已用 > WARN_SWAP_MB 时为 1，
# 便于 cron 邮件告警。
set -uo pipefail

LOG_FILE="${MEM_WATCH_LOG:-/home/ubuntu/logs/mem-watch.log}"
MAX_LOG_MB="${MEM_WATCH_MAX_LOG_MB:-8}"
WARN_AVAILABLE_MB="${MEM_WATCH_WARN_AVAILABLE_MB:-500}"
WARN_SWAP_MB="${MEM_WATCH_WARN_SWAP_MB:-512}"

to_stdout=0
[[ "${1:-}" == "--stdout" ]] && to_stdout=1

# 各容器 cgroup 的真实占用。docker stats 的数字含容器内页面缓存，
# 排查时需要 anon 才有意义。
container_rows() {
  local cid name pid cg dir cur anon
  for cid in $(docker ps -q 2>/dev/null); do
    name=$(docker inspect -f '{{.Name}}' "$cid" 2>/dev/null | tr -d /)
    pid=$(docker inspect -f '{{.State.Pid}}' "$cid" 2>/dev/null)
    [[ -r "/proc/$pid/cgroup" ]] || continue
    cg=$(grep -o '0::.*' "/proc/$pid/cgroup" 2>/dev/null | cut -d: -f3)
    dir="/sys/fs/cgroup${cg}"
    cur=$(cat "$dir/memory.current" 2>/dev/null) || continue
    anon=$(awk '/^anon /{print $2}' "$dir/memory.stat" 2>/dev/null)
    printf '%s\t%s\t%s\n' "$name" "${cur:-0}" "${anon:-0}"
  done | awk -F'\t' '{printf "  %-24s %8.0f MB  (anon %6.0f MB / file %6.0f MB)\n", $1, $2/1048576, $3/1048576, ($2-$3)/1048576}' \
    | sort -k2 -rh
}

snapshot() {
  local line
  line=$(LC_ALL=C date '+%Y-%m-%d %H:%M:%S')
  echo "===== $line ====="
  free -m | sed -n '1,2p' | sed 's/^/  /'
  echo "  --- 容器（总占用 / anon 真实 / file 缓存）---"
  container_rows
  echo
}

snapshot

if [[ $to_stdout -eq 1 ]]; then
  exit 0
fi

mkdir -p "$(dirname "$LOG_FILE")"

# 简单防抖：超过上限就截断一半，避免日志无限增长把磁盘吃满
if [[ -f "$LOG_FILE" ]]; then
  local_mb=$(( $(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0) / 1048576 ))
  if (( local_mb > MAX_LOG_MB )); then
    tail -n 2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
  fi
fi

snapshot >> "$LOG_FILE"

# 告警判定
read -r _ total used free shared buffcache available _ < <(free -m | sed -n '2p')
swap_used_kb=$(awk '/^SwapTotal/{t=$2} /^SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
rc=0
if (( available < WARN_AVAILABLE_MB )); then
  echo "WARN: 可用内存仅 ${available}MB（阈值 ${WARN_AVAILABLE_MB}MB）"
  rc=1
fi
swap_used_mb=$(( swap_used_kb / 1024 ))
if (( swap_used_mb > WARN_SWAP_MB )); then
  echo "WARN: swap 已用 ${swap_used_mb}MB（阈值 ${WARN_SWAP_MB}MB）"
  rc=1
fi
exit $rc
