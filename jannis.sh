#!/usr/bin/env bash

# SET YOUR ENVIRONMENT HERE
: "${MEMORY_LIMIT:=8}"
: "${GPUS:?Must set GPUS (or 'none')}"
: "${LOCAL_CODE:?You must set your code directory which will be mounted}"
: "${LOCAL_DATA:?You must set the data directory}"
: "${LOCAL_HOME:="${HOME}"}"

# You shouldn't need to modify anything later than here.
repo="janniskatis/nlp_image"

memory_limit="${MEMORY_LIMIT}G"

remote_home="/home/user"
remote_data="/data"

local_pretrained="${LOCAL_DATA}/pretrained"
mkdir -p "${local_pretrained}"

docker=(
    "docker"
    "run"
    "--rm"
    "--init"
    "-it"
    "--memory" "${memory_limit}"
    "--memory-swap" "${memory_limit}"
    "--shm-size=${memory_limit}"
    "--ulimit" "memlock=-1"
    "--ulimit" "stack=67108864"
    "-e" "TF_FORCE_GPU_ALLOW_GROWTH=true"
    "-v" "${LOCAL_CODE}:${remote_home}/code"
    "-v" "${LOCAL_DATA}:${remote_data}"
    "-w" "${remote_home}/code"
 )

# Run Jupyter if a PORT is set, else run bash
if [ -z "${PORT+x}" ];
then
    inner=( "bash" )
else
    docker+=("-p" "127.0.0.1:${PORT}:${PORT}")
    inner=( "jupyter" "lab" "--ip" "*" "--port" "${PORT}" )
fi

# Overwrite inner command if there are commands
if [ ! $# -eq 0 ]; then
    inner=( "$@" )
fi

docker+=(
  "${nemo_image}"
  "${inner[@]}"
)

exec "${docker[@]}"
