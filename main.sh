sudo GPUS="0" MEMORY_LIMIT="32" LOCAL_CODE="/home/mcgrau/PycharmProjects/NLP_SS_22" LOCAL_DATA="/data" PORT="8888" ./start_env.sh

python3 train.py \
  --config-path=configs/models/nemo \
  --config-name=conformerCTC

