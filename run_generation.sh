PRETRAINDIR=microsoft/CodeGPT-small-py

python code/run_generation.py \
       --model_type=gpt2 \
       --model_name_or_path=$PRETRAINDIR \
       --k=40 \
       --temperature=0.8 \
