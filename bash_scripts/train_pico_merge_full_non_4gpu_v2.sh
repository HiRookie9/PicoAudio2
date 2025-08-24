accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml \
    train.py \
    epochs=200 \
    data@data_dict=merge_data_max_new \
    train_dataloader.batch_size=8 \
    val_dataloader.batch_size=4 \
    model=diffusion_pico_merge_full_non \
    exp_dir=exp/diffusion_pico_merge_full_non_v2_4gpu_bs8 \
    trainer.wandb_config.project=diffusion_pico_merge_full_non_v2_4gpu_bs8 \
    +trainer.resume_from_checkpoint=/hpc_stor03/sjtu_home/zihao.zheng/x_to_audio_generation/exp/diffusion_pico_merge_full_non_v2_4gpu_bs8/checkpoints/best
    
