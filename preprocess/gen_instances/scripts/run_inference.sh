NUM_GPU=8
NUM_PROBLEM=15360
NUMBER_PER_PROCESS=$((NUM_PROBLEM/NUM_GPU))

CUDAid=(0 1 2 3 4 5 6 7)

date

JOB_NAME=''
MODEL_PATH=''
DATA_PATH=''
RESULT_PATH=''
TASK='' # qa, math, align

for ((i=0;i<$NUM_GPU;i++)) do
{
    startidx=$(((i*NUMBER_PER_PROCESS)))
	endidx=$((((i+1)*NUMBER_PER_PROCESS)))
    cuda0idx=$i
    rseed=$((i+2023))
    echo $startidx $endidx
    python inference.py \
        --start_idx $startidx \
        --end_idx $endidx \
        --batch_size 16 \
        --model_path ${MODEL_PATH} \
        --data_name ${TASK} \
        --data_path ${DATA_PATH}  \
        --target_path ${RESULT_PATH}/result-$startidx-$endidx.jsonl \
        --cuda_device ${CUDAid[$cuda0idx]} \
        --write_mode w \
        --seed $rseed \
    &> logs/${JOB_NAME}/part$i.log
}&
done

wait

date
