cd ../tools/

mkdir -p ../data/HJDataset/annotations/

python process_HJDataset.py \
    --origpath ../data/HJDataset/raw/annotations/main_train.json \
    --savepath ../data/HJDataset/annotations/train.json \
    --selected_categories 4 5 6 7 

echo "[1/2] The train set for HJDataset has been processed!"
echo "======================================"

python process_HJDataset.py \
    --origpath ../data/HJDataset/raw/annotations/main_val.json \
    --savepath ../data/HJDataset/annotations/val.json \
    --selected_categories 4 5 6 7 
echo "[2/2] The val set for HJDataset has been processed!"
echo "======================================"