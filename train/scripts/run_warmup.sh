export WANDB_MODE=offline
export OMP_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


DATA_PATH=''
BACKBONE_MODEL=''
CKPT=''

JOB_NAME=''
SAVE_DIR=''

torchrun --nproc_per_node=8 \
    --master_port=23412 \
    train_warmup.py \
    --model_name_or_path ${BACKBONE_MODEL} \
    --dataset_path ${DATA_PATH} \
    --output_dir ${SAVE_DIR}/${JOB_NAME} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --seed 2023 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --max_source_length 1200 \
    --max_target_length 500 \
    --max_length 1200 \
    --warmup_ratio 0.1 \
    --save_strategy epoch \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --use_legacy_prediction_loop \
    --label_names labels \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --deepspeed configs/ds_z3_bf16.json \
&> logs/${JOB_NAME}.log

cd ${SAVE_DIR}/${JOB_NAME}/checkpoint-${CKPT}
python zero_to_fp32.py ./ ./pytorch_model.bin
cd ~/ALLO-code