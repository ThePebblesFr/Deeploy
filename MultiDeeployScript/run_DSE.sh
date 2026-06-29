#!/bin/bash

# ---- CONFIG ----
DOCKER_CONTAINER="deeploy_main"

TESTS=(
  "Tests/testRQConv"
)

CORES_LIST=(1 2 3 4 5 6 7 8)
L1_LIST=($(seq 4096 4096 65536))

BASE_DIR="/home/pierre/Documents/thesis/gitlab-repo/projects/Deeploy-pga/DeeployTest/TEST_SIRACUSA/build"

WORKDIR="$BASE_DIR/gvsoc_workdir"
BIN_DIR="$BASE_DIR/bin"

# ---- LOOP ----
for test in "${TESTS[@]}"; do

	# Normalize path (remove trailing slash)
	test=${test%/}

	# Extract model name (last folder)
	MODEL_NAME=$(basename "$test")

	# Binary assumed to match model name
	BINARY="$BIN_DIR/$MODEL_NAME"

	# Output folder
	OUTPUT_DIR="gvsoc_log_${MODEL_NAME}"
	mkdir -p "$OUTPUT_DIR"

	echo "========================================"
	echo "Starting model: $MODEL_NAME"
	echo "Test path: $test"
	echo "Binary: $BINARY"
	echo "Logs: $OUTPUT_DIR"
	echo "========================================"

	for cores in "${CORES_LIST[@]}"; do
		for l1 in "${L1_LIST[@]}"; do

			echo "--- cores=$cores l1=$l1 ---"

			# ---- 1. Docker ----
			docker exec "$DOCKER_CONTAINER" bash -c "
			cd /app/Deeploy/DeeployTest && \
			python3 testRunner_tiled_siracusa.py \
			-t $test \
			--cores $cores \
			--l1 $l1
			"

			DOCKER_STATUS=$?

			if [ $DOCKER_STATUS -ne 0 ]; then
			echo "⚠️ Docker failed (code=$DOCKER_STATUS), continuing..."
			fi

			# ---- 2. Run on host ----
			LOG="${OUTPUT_DIR}/gvsoc_cores${cores}_l1${l1}.log"

			if gvsoc \
				--target=siracusa \
				--work-dir="$WORKDIR" \
				--binary="$BINARY" \
				--power image flash run \
				> "$LOG" 2>&1
			then
				echo "✅ Saved: $LOG"
			else
				echo "❌ gvsoc failed (see $LOG)"
				echo "gvsoc failed" > "$LOG"
			fi

			# ---- 3. Remove binary ----
			if [ -f "$BINARY" ]; then
				rm -f "$BINARY"
				echo "Removed binary: $BINARY"
			else
				echo "Binary not found: $BINARY"
			fi

		done
	done
done