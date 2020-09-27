cd ../tools/

python cococv.py \
    --annotation_path     ../data/prima/annotations/all.json \
    --save_path           ../data/prima/annotations/cv \
    --folds               5 \
    --having-annotations

echo "The prima dataset has splitted into training and testing set for 5 folds!"
echo "======================================"