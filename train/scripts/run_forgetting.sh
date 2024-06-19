export WANDB_MODE=offline
export OMP_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


MASK_PATH=''
DATA_PATH=''
BACKBONE_MODEL=''
CKPT=''

JOB_NAME=''
SAVE_DIR=''

torchrun --nproc_per_node=8 \
    --master_port=12345 \
    train_forgetting.py \
    --model_name_or_path ${BACKBONE_MODEL} \
    --data_path ${DATA_PATH} \
    --mask_path ${MASK_PATH} \
    --bf16 True \
    --output_dir ${SAVE_DIR}/${JOB_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --deepspeed configs/ds_z3_bf16.json \
    --gradient_checkpointing True \
    --tf32 True \
&> logs/${JOB_NAME}.log

cd ${SAVE_DIR}/${JOB_NAME}/checkpoint-${CKPT}
python zero_to_fp32.py ./ ./pytorch_model.bin
cd ~/ALLO-code


