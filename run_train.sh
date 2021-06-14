LANG=java
DATADIR=./dataset/javaCorpus/token_completion
OUTPUTDIR=./save/javaCorpus
PRETRAINDIR=./pretrained/microsoft/CodeGPT-small-java
LOGFILE=./logs/completion_javaCorpus_train.log

python -u ./code/run_lm.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --block_size=1024 \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --num_train_epochs=2 \
        --per_gpu_train_batch_size=2 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --logging_steps=100 \
        --save_steps=100 \
        --seed=42 \
        --evaluate_during_training \
        --overwrite_output_dir \
        --use_pretrain \
        --do_train