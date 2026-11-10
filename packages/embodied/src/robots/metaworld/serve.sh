#!/usr/bin/env bash
# Start (default) or stop the Pi0.5 server of the Metaworld robot's VLA alone (--vla-alone pi0):
# RLinf's Pi0.5 MetaWorld SFT, served with the Metaworld embodiment at services.metaworld_vla.
#   [PI_EMBODIED_SERVICES=...] PI_EMBODIED_PYTHON=... PI05_METAWORLD_CHECKPOINT_PATH=... serve.sh [start|stop]
# `services/setup.sh metaworld --weights` installs the server's dependencies into the Metaworld venv,
# downloads the checkpoint and writes both variables to <venv>/pi-embodied.env.
# start records the server it launches in $LOG_DIR/<module>-<port>.pid, and stop stops only that one:
# a port that already answers is left alone, so a server someone else started keeps running.
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../../services" && pwd)}
PY=${PI_EMBODIED_PYTHON:-python}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${METAWORLD_VLA_PORT:-18250}
VLA=pi_embodied_services.components.pi05_vla_server
mkdir -p "$LOG_DIR"

up() { curl -sf -X POST "http://127.0.0.1:$1/call" -d '{"method":"healthz"}' | grep -q '"ok": *true'; }

if [ "${1:-start}" = stop ]; then
	f="$LOG_DIR/$VLA-$VLA_PORT.pid"
	[ -f "$f" ] || exit 0
	pid=$(cat "$f")
	if ps -o command= -p "$pid" | grep -q -- "$VLA"; then
		kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
		for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
		kill -9 -- "-$pid" 2>/dev/null || true
	fi
	rm -f "$f"
	exit 0
fi

: "${PI05_METAWORLD_CHECKPOINT_PATH:?set PI05_METAWORLD_CHECKPOINT_PATH (services/setup.sh metaworld --weights downloads RLinf/RLinf-Pi05-MetaWorld-SFT)}"
cd "$SERVICES"
export PYTHONPATH=$SERVICES
if up "$VLA_PORT"; then
	echo "already serving on :$VLA_PORT (not started here; stop leaves it running)"
else
	setsid nohup "$PY" -m "$VLA" --embodiment metaworld --transport http --model-path "$PI05_METAWORLD_CHECKPOINT_PATH" \
		--host 127.0.0.1 --port "$VLA_PORT" >"$LOG_DIR/metaworld_vla_server.log" 2>&1 </dev/null &
	echo $! >"$LOG_DIR/$VLA-$VLA_PORT.pid"
fi
for _ in $(seq 600); do
	up "$VLA_PORT" && break
	sleep 1
done
up "$VLA_PORT" || { echo "server on :$VLA_PORT not ready; see $LOG_DIR/metaworld_vla_server.log" >&2; exit 1; }
echo "ready on :$VLA_PORT"
