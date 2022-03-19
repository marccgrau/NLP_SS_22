#!/usr/bin/env bash

# SET YOUR ENVIRONMENT HERE
: "${MEMORY_LIMIT:=8}"
: "${GPUS:?Must set GPUS (or 'none')}"
: "${LOCAL_CODE:?You must set your code directory which will be mounted}"
: "${LOCAL_DATA:?You must set the data directory}"
: "${LOCAL_HOME:="${HOME}"}"

# You shouldn't need to modify anything later than here.
repo="registry.gitlab.com/ds-unisg/servers/python-docker"

python="39"
pytorch="110"
nemo="17"

if [ "${GPUS}" == "none" ];
then
  arch="cpu"
else
  arch="gpu"
fi

NEMO_IMAGE="${repo}/py${python}/${arch}-ds-pytorch${pytorch}-nemo${nemo}"

if [ -z "${PORT+x}" ];
then
    nemo_image="${NEMO_IMAGE}:cli"
else
    nemo_image="${NEMO_IMAGE}:jupyter"
fi

if [ -z "$(docker images -q "${nemo_image}")" ];
then
    echo "This image does not exist: ${nemo_image}"
    echo "You could try to pull it:"
    echo "docker pull ${nemo_image}"
    exit 1
fi

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
    "-v" "${LOCAL_HOME}/.netrc:${remote_home}/.netrc:ro"
    "-v" "${LOCAL_CODE}:${remote_home}/code"
    "-v" "${LOCAL_DATA}:${remote_data}"
    "-v" "${local_pretrained}:${remote_home}/.cache/torch/NeMo"
    "-w" "${remote_home}/code"
 )

# Add GPU if they are set
if [ "${GPUS}" != "none" ];
then
    docker+=(
        "--gpus=\"device=${GPUS}\""
    )
fi

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
