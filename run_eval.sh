LANG=java
DATADIR=./dataset/javaCorpus/token_completion
OUTPUTDIR=./save/javaCorpus
PRETRAINDIR=./pretrained/microsoft/CodeGPT-small-java
LOGFILE=./logs/completion_javaCorpus_eval.log

python -u ./code/run_lm.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --block_size=1024 \
        --per_gpu_eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42 \
        --do_eval