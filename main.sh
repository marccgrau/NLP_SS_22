GPUS="0"
MEMORY_LIMIT="32"
LOCAL_CODE="/home/mcgrau/PycharmProjects/swiss2text"
LOCAL_DATA="/data"
./start_env.sh

python3 trainConfig.py \
  --config-path=configs/models/nemo \
  --config-name=conformerTD