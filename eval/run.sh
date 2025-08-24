vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-zzh_pico:0.0.3 \
    --partition pdgpu-a10 \
    --job xse_pico_eval \
    --num-task 1 \
    --cpu-per-task 4 \
    --mem-per-task 24G \
    --gpu-per-task 1 \
    --cmd "python /hpc_stor03/sjtu_home/zihao.zheng/RF_DiT/run.py"