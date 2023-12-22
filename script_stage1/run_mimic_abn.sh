#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=false
export WANDB_PROJECT="Recap"
suffix=""
warmup_ratio=0.0
max_tgt_length=64
num_train_epochs=10
overwrite_output_dir=false
evaluation_strategy=epoch
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=2
debug_model=false
seed=42
num_beams=4
slow_lr=1e-4
fast_lr=1e-4
weight_decay=0.0
alpha=2
beta=1
dataloader_num_workers=8
log_level="info"
report_to="wandb"
chexbert_label="./CheXbert/mimic_abn/id2tag.csv"
annotation_file="./mimic_abn/annotation.json"
miss_annotation_file="./mimic_abn/ref_annotation.json" # collected from the original annotation file, same format of the annotation_file
output_dir="./tmp_stage1/mimic_abn_stage1/"

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    suffix="_debug"
    num_train_epochs=1
    output_dir="./tmp/toy_debug"
    overwrite_output_dir=true
    debug_model=true
    report_to="none"
fi

export TOKENIZERS_PARALLELISM=true
python3 -u ./src_stage1/run_ende.py \
    --chexbert_model_name_or_path ../CheXbert/chexbert.pth \
    --annotation_file $annotation_file \
    --miss_annotation_file $miss_annotation_file \
    --history "../mimic_abn/temporal_ids.json" \
    --image_path ../mimic_cxr/images/ \
    --chexbert_label $chexbert_label \
    --do_train \
    --do_eval \
    --do_predict \
    --log_level $log_level \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --max_tgt_length $max_tgt_length \
    --output_dir $output_dir \
    --warmup_ratio $warmup_ratio \
    --num_train_epochs $num_train_epochs \
    --learning_rate $slow_lr \
    --fast_lr $fast_lr \
    --weight_decay $weight_decay \
    --evaluation_strategy $evaluation_strategy \
    --save_strategy $evaluation_strategy \
    --save_total_limit 1 \
    --alpha $alpha \
    --beta $beta \
    --seed $seed \
    --logging_steps 100 \
    --report_to $report_to \
    --fp16 \
    --fp16_opt_level O2 \
    --fp16_full_eval \
    --dataloader_num_workers $dataloader_num_workers \
    --load_best_model_at_end true \
    --overwrite_output_dir $overwrite_output_dir \
    --group_by_length false \
    --length_column_name length \
    --eval_on_gen \
    --greater_is_better true \
    --metric_for_best_model eval_macro_f1 \
    --debug_model $debug_model