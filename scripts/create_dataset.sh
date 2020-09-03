cd ../tools/

python convert_prima_to_coco.py \
    --prima_datapath ../data/prima/raw \
    --anno_savepath  ../data/prima/annotations/all.json 
echo "[1/4] The prima dataset has been converted to the COCO format!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/prima/annotations/all.json \
    --split_ratio         0.8 \
    --train               ../data/prima/annotations/train.json \
    --test                ../data/prima/annotations/val.json \
    --having-annotations
echo "[2/4] The prima dataset has splitted into training and testing set!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/publaynet/raw/annotations/val.json \
    --split_ratio         0.8 \
    --train               ../data/publaynet/annotations/train-sml.json \
    --test                ../data/publaynet/annotations/val-sml.json \
    --having-annotations
echo "[3/4] The publaynet dataset has splitted into training and testing set!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/coco/raw/annotations/instances_val2017.json \
    --split_ratio         0.8 \
    --train               ../data/coco/annotations/train-sml.json \
    --test                ../data/coco/annotations/val-sml.json \
    --having-annotations
# Used for adding a dummy background class in the original train and val files 
python cocosplit.py \
    --annotation_path     ../data/coco/raw/annotations/instances_val2017.json \
    --split_ratio         1 \
    --train               ../data/coco/annotations/val.json \
    --test                ../data/coco/annotations/tmp.json \
    --having-annotations

python cocosplit.py \
    --annotation_path     ../data/coco/raw/annotations/instances_train2017.json \
    --split_ratio         1 \
    --train               ../data/coco/annotations/train.json \
    --test                ../data/coco/annotations/tmp.json \
    --having-annotations
echo "[4/4] The coco dataset has splitted into training and testing set!"
echo "======================================"