export CUDA_VISIBLE_DEVICES=0,1,2,3


L_PER=0
R_PER=5

REF_MODEL_PATH=''
TRAINED_MODEL_PATH=''
RESULT_PATH=''

python gen_neuron_mask.py \
    --ref_model_path ${REF_MODEL_PATH}/pytorch_model.bin \
    --trained_model_path ${TRAINED_MODEL_PATH}/pytorch_model.bin \
    --result_path ${RESULT_PATH}/lper${L_PER}_rper${R_PER}.bin \
    --l_per ${L_PER} \
    --r_per ${R_PER} \
    --is_neg 1
