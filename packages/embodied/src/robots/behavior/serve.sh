#!/usr/bin/env bash
# Start (default) or stop the Pi0.5 server of the BEHAVIOR-1K robot's VLA alone (--vla-alone pi0):
# RLinf's Pi0.5 BEHAVIOR-1K checkpoint, served with the behavior embodiment at services.behavior_vla.
#   [PI_EMBODIED_SERVICES=...] PI05_PYTHON=... PI05_BEHAVIOR_CHECKPOINT_PATH=... serve.sh [start|stop]
# The server needs RLinf and its openpi, which the BEHAVIOR-1K venv (Isaac Sim, Python 3.12) does not carry:
# PI05_PYTHON is the python of a venv that does (the LIBERO one, or the Metaworld one after
# `services/setup.sh metaworld --weights`). `services/setup.sh behavior --weights` downloads the checkpoint.
# On an sm_120 GPU (RTX 50 series, RTX PRO 6000 Blackwell) that venv needs torch==2.7.1+cu128: RLinf's
# cu126 pin loads the model and then fails the first inference ("no kernel image is available").
# start records the server it launches in $LOG_DIR/<module>-<port>.pid, and stop stops only that one:
# a port that already answers is left alone, so a server someone else started keeps running.
set -euo pipefail
SERVICES=${PI_EMBODIED_SERVICES:-$(cd "$(dirname "$0")/../../../../../services" && pwd)}
LOG_DIR=${LOG_DIR:-/tmp/pi-embodied}
VLA_PORT=${BEHAVIOR_VLA_PORT:-18260}
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

: "${PI05_PYTHON:?set PI05_PYTHON to the python of a venv with RLinf and openpi (the LIBERO venv, or the Metaworld one after setup.sh metaworld --weights)}"
: "${PI05_BEHAVIOR_CHECKPOINT_PATH:?set PI05_BEHAVIOR_CHECKPOINT_PATH (services/setup.sh behavior --weights downloads RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32)}"
cd "$SERVICES"
export PYTHONPATH=$SERVICES
if up "$VLA_PORT"; then
	echo "already serving on :$VLA_PORT (not started here; stop leaves it running)"
else
	setsid nohup "$PI05_PYTHON" -m "$VLA" --embodiment behavior --transport http --model-path "$PI05_BEHAVIOR_CHECKPOINT_PATH" \
		--host 127.0.0.1 --port "$VLA_PORT" >"$LOG_DIR/behavior_vla_server.log" 2>&1 </dev/null &
	echo $! >"$LOG_DIR/$VLA-$VLA_PORT.pid"
fi
for _ in $(seq 600); do
	up "$VLA_PORT" && break
	sleep 1
done
up "$VLA_PORT" || { echo "server on :$VLA_PORT not ready; see $LOG_DIR/behavior_vla_server.log" >&2; exit 1; }
echo "ready on :$VLA_PORT"
