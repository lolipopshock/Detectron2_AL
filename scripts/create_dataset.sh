cd ../tools/

python convert_prima_to_coco.py \
    --prima_datapath ../data/prima \
    --anno_savepath  ../data/prima/annotations.json 
echo "[1/4] The prima dataset has been converted to the COCO format!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/prima/annotations.json \
    --split_ratio         0.8 \
    --train               ../data/prima/train.json \
    --test                ../data/prima/val.json \
    --having-annotations
echo "[2/4] The prima dataset has splitted into training and testing set!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/publaynet/annotations/val.json \
    --split_ratio         0.8 \
    --train               ../data/publaynet/annotations/train-sml.json \
    --test                ../data/publaynet/annotations/val-sml.json \
    --having-annotations
echo "[3/4] The publaynet dataset has splitted into training and testing set!"
echo "======================================"

python cocosplit.py \
    --annotation_path     ../data/coco/annotations/instances_val2017.json \
    --split_ratio         0.8 \
    --train               ../data/coco/annotations/train-sml.json \
    --test                ../data/coco/annotations/val-sml.json \
    --having-annotations
echo "[4/4] The coco dataset has splitted into training and testing set!"
echo "======================================"