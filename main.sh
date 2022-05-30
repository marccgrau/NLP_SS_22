sudo GPUS="0" MEMORY_LIMIT="32" LOCAL_CODE="/home/mcgrau/PycharmProjects/NLP_SS_22" LOCAL_DATA="/data" PORT="8888" ./start_env.sh

python3 train.py \
  --config-path=configs/models/nemo \
  --config-name=conformerCTC

ffmpeg -i 00-KR-Sitzung_2021-02-24-STE-000.wav -f segment -segment_time 30 -c copy parts/output%09d.wav ## split audio into 30s segments