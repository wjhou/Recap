version="20230901"
dataset="mimic_abn"
min_count=200
max_count=5000000
chexbert_label="../CheXbert/src/data/$dataset/id2tag_ref_64.csv"
output_dir="data/$version/"
temporal_id_dir="../$dataset/temporal_ids.json"

mkdir -p $output_dir$dataset

echo "================================================================"
echo "Step1: running python src_stage1/graph_construction/pmi_ngram.py"
echo "================================================================"
python src_stage1/graph_construction/pmi_ngram.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --min_count $min_count

echo "================================================================"
echo "Step 2: running python src_stage1/graph_construction/pmi_observation_ngram.py"
echo "================================================================"
python src_stage1/graph_construction/pmi_observation_ngram.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --pmi_threshold 0 \
    --min_count $min_count \
    --max_count $max_count
    
echo "================================================================"
echo "Step 3: running python src_stage1/graph_construction/pmi_progression_ngram.py"
echo "================================================================"
python src_stage1/graph_construction/pmi_progression_ngram.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --pmi_threshold 0 \
    --min_count $min_count \
    --temporal_id_dir $temporal_id_dir