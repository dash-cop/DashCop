
# conda activate wsd_main

python train_lightning.py --wandb  2>&1 | tee train_lightning.log

python train_lightning.py --wandb --wandb_name "weigthed_sam3_puf" \
    --resume /ssd_scratch/sai.teja/triple_violations/checkpoints_ogsplit_sam3/weighted/tr_clf_sam3.ckpt \
    2>&1 | tee train_lightning.log