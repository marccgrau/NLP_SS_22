GPUS="0" MEMORY_LIMIT="32" LOCAL_CODE="/home/mcgrau/PycharmProjects/NLP_SS_22" LOCAL_DATA="/data" ./start_env.sh

python3 trainConfig.py \
  --config-path=configs/models/nemo \
  --config-name=conformerCTC

