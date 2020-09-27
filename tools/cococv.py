# Modified based on https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py

import json
import argparse
import funcy
import random
import os 

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and val sets.')
parser.add_argument('--annotation_path', type=str, help='Path to COCO annotations file.')
parser.add_argument('--save_path',       type=str, help='Where to store COCO generated annotations.')
parser.add_argument('--folds',           type=int, required=True, help="The number of folds.")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(annotation_path,
         save_path,
         folds,
         having_annotations,
         random_state=None):

    random.seed(random_state)
    
    with open(annotation_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info', '')
        licenses = coco.get('licenses', '')
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

    if having_annotations:
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    num_images = len(images)
    fold_size  = num_images // folds

    image_indices = list(range(num_images))
    random.shuffle(image_indices)

    for fold in range(folds):

        val_indices = list(range(fold_size*(fold), fold_size*(fold+1)))

        train = [images[idx] for idx in image_indices if idx not in val_indices]
        val  = [images[idx] for idx in image_indices if idx in val_indices]

        os.makedirs(f'{save_path}/{fold}')
        train_save_path = f'{save_path}/{fold}/train.json'
        val_save_path = f'{save_path}/{fold}/val.json'
        
        save_coco(train_save_path, info, licenses, train, filter_annotations(annotations, train), categories)
        save_coco(val_save_path, info, licenses, val, filter_annotations(annotations, val), categories)

        print("[Fold {}] Saved {} entries in {} and {} in {}".format(fold, len(train), train_save_path, len(val), val_save_path))


if __name__ == "__main__":
    args = parser.parse_args()

    main(args.annotation_path,
         args.save_path,
         args.folds,
         args.having_annotations, 
         random_state=42)