export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=ERROR
export ATB_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.98
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"

\rm -rf /root/atb/log/
\rm -rf /root/ascend/log/
\rm -rf core.*

MODEL_PATH="/export/home/yinjiawei.15/models/Qwen3-4B/"
# MASTER_NODE_ADDR="127.0.0.1:9590"
MASTER_NODE_START_PORT=9590
START_PORT=14830
START_DEVICE=5
LOG_DIR="log"
NNODES=2
WORLD_SIZE=1

export HCCL_IF_BASE_PORT=43439

  for (( i=0; i<$NNODES; i++ ))
do
  MASTER_NODE_PORT=$((MASTER_NODE_START_PORT + i))
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr="127.0.0.1:$MASTER_NODE_PORT" \
    --nnodes=$WORLD_SIZE \
    --devices="npu:$DEVICE" \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=20000 \
    --max_seqs_per_batch=3000 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="lccl" \
    --enable_schedule_overlap=false \
    --enable_mla=false \
    --dp_size=1 \
    --ep_size=1 \
    --double_weights_buffer=true \
    --train_mode=$([ $i -eq 0 ] && echo "true" || echo "false") \
    --device_id=$DEVICE \
    --rank_id=$i \
    --rank_size=$NNODES \
    --remote_rank_id=$(( (i + 1) % NNODES )) \
    > $LOG_FILE 2>&1 &
done

