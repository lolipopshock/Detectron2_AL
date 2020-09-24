import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--origpath', type=str, default='')
parser.add_argument('--savepath', type=str, default='')
parser.add_argument('--selected_categories', type=int, nargs='+', default=[])

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

def write_json(res, filename):
    with open(filename, 'w') as fp:
        json.dump(res, fp)

if __name__ == "__main__":
    args = parser.parse_args()
    
    coco = load_json(args.origpath)
    cats = args.selected_categories
    annos = [ele for ele in coco['annotations'] if ele['category_id'] in cats]
    coco['annotations'] = annos

    write_json(coco, args.savepath)