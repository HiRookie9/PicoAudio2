accelerate launch --config_file configs/accelerate/nvidia/4gpus.yaml \
    train.py \
    epochs=50 \
    data@data_dict=example \
    train_dataloader.batch_size=8 \
    val_dataloader.batch_size=4 \
    model=diffusion \
    exp_dir=exp/diffusion_4gpu \
    trainer.wandb_config.project=diffusion_4gpu \
    #+trainer.resume_from_checkpoint=exp/diffusion_4gpu/checkpoints/best
    
