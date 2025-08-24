vc submit --image docker.v2.aispeech.com/sjtu/sjtu_wumengyue-zzh_x2audio_pico:0.0.2 \
    --partition pdgpu-a10 \
    --nopassenv \
    --env HF_HOME="/hpc_stor03/sjtu_home/xuenan.xu/hf_cache" TOKENIZERS_PARALLELISM=false \
    --job x2audio_pico_4gpu \
    --num-task 1 \
    --cpu-per-task 16 \
    --mem-per-task 96G \
    --gpu-per-task 4 \
    JOB=1:1 logs/vc/without_onset.JOB.log \
    --cmd "bash bash_scripts/train_pico_merge_full_non_4gpu_v2.sh"