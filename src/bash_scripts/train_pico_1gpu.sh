accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
    train.py \
    epochs=50 \
    data@data_dict=example \
    train_dataloader.batch_size=8 \
    val_dataloader.batch_size=4 \
    model=diffusion \
    exp_dir=exp/diffusion\
    trainer.wandb_config.project=diffusion \
    #+trainer.resume_from_checkpoint=exp/diffusion/checkpoints/best