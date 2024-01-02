version="2023xxxx"
dataset="mimic_abn"
radgraph_dir="radgraph/1.0.0/MIMIC-CXR_graphs.json"
min_count=200
chexbert_label="./CheXbert/$dataset/id2tag.csv"
output_dir="data/$version/"
temporal_id_dir="../$dataset/temporal_ids.json"

mkdir -p $output_dir$dataset

echo "================================================================"
echo "Step1: running python src_stage1/graph_construction/prepare_stat.py"
echo "================================================================"
python src_stage1/graph_construction/prepare_stat.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --radgraph_dir $radgraph_dir

echo "================================================================"
echo "Step 2: running python src_stage1/graph_construction/pmi_observation_entity.py"
echo "================================================================"
python src_stage1/graph_construction/pmi_observation_entity.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --pmi_threshold 0 \
    --min_count $min_count
    
echo "================================================================"
echo "Step 3: running python src_stage1/graph_construction/pmi_progression_entity.py"
echo "================================================================"
python src_stage1/graph_construction/pmi_progression_entity.py \
    --dataset $dataset \
    --chexbert_label $chexbert_label \
    --output_dir $output_dir \
    --pmi_threshold 0 \
    --min_count $min_count \
    --temporal_id_dir $temporal_id_dir