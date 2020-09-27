import json
import os 
import sys
from glob import glob
from itertools import cycle 
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

sys.path.append('../src')
from detectron2_al.configs import get_cfg


AP_FEATURES = ['AP', 'AP50', 'AP75', 'APl', 'APm', 'APs']
LB_FEATURES = ['num_images', 'num_objects', 'training_iter']

def load_cfg(config_file_path):
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)
    return cfg

def load_json(json_path):
    with open(json_path, 'r') as fp:
        return json.load(fp)

def create_color_palette(show=False):
    object_plot_color = sns.color_palette("GnBu_d")
        #sns.light_palette((210, 90, 60), input="husl")
    image_plot_color = sns.color_palette("RdPu")
        #sns.light_palette((45, 90, 60), input="husl")
    
    if show:
        sns.palplot(object_plot_color)
        sns.palplot(image_plot_color)
    
    return {"object": object_plot_color[1:], "image":image_plot_color[1:]}


class Experiment:

    def __init__(self, name, 
                    base_path,
                    config_name = 'config.yaml',
                    history_name = 'labeling_history.json',
                    evaluation_folder = 'evals',
                    allow_unfinished = True):
        """For a single run of a dataset experiment.
        """
        self.name = name

        self.base_path = base_path  
        self.cfg = load_cfg(os.path.join(self.base_path, config_name))
        self.history_dir = os.path.join(self.base_path, self.cfg.AL.DATASET.CACHE_DIR)
        self.eval_dir = os.path.join(self.base_path, evaluation_folder)

        self.evals = self.load_all_evals()
        self.labeling_history = load_json(os.path.join(self.base_path, self.cfg.AL.DATASET.CACHE_DIR, history_name))

    def load_all_evals(self):
        num_all_evals = len(glob(os.path.join(self.eval_dir,'*.csv')))
        
        # Ensure the evals are ordered correctly
        # Assuming the experiment start from 0 to num_all_evals-1
        return [
            pd.read_csv(os.path.join(self.eval_dir, f'{idx}.csv'), index_col=0) for idx in range(num_all_evals)
        ]   


    def load_training_stats(self):

        res = []
    
        for idx, (eval_res, history) in enumerate(zip(self.evals, self.labeling_history)):

            ap_info = eval_res.loc[AP_FEATURES,'bbox'].to_list()   
            labeling_info = [history[feat] for feat in LB_FEATURES]
            res.append([idx] + ap_info + labeling_info)

        df = pd.DataFrame(res, columns=['round']+ AP_FEATURES + LB_FEATURES)

        for feat in LB_FEATURES:
            df[f'cum_{feat}'] = df[feat].cumsum()

        return df
        

    def load_history_dataset_acc(self):
        res = []
        for idx, filename in enumerate(sorted(glob(self.history_dir + '/*eval.csv'))):
            ap_score = pd.read_csv(filename, index_col=0, header=None).loc['AP'].iloc[0]
            res.append([idx+1, ap_score])
        res = pd.DataFrame(res, columns=['round', 'AP'])
        return res


class Experiments:

    def __init__(self, base_path,
                       dataset_name,
                       architecture_name  = 'faster_rcnn_R_50_FPN'):
        
        """For all runs within experiments for a dataset.
        """

        self.base_path = base_path
        self.dataset_name = dataset_name
        self.architecture_name = architecture_name

        self.exps = {}
        for exp_cat in ['image', 'object']:
            self.exps[exp_cat] = self.load_experiments(f'{self.base_path}/{exp_cat}/{self.architecture_name}')

    @staticmethod
    def load_experiments(base_path):
        
        all_exps = []
        for name in os.listdir(base_path):
            try:
                exp = Experiment(name = name, base_path = os.path.join(base_path, name))
                all_exps.append(exp)
            except:
                print(f"skipped for {name}")
        return all_exps


    def plot_training_stats(self, xaxis='cum_num_objects', yaxis='AP'):

        names = []

        color_maps = create_color_palette()

        for exp_cat in ['image', 'object']:
            for (exp, color) in zip(self.exps[exp_cat], cycle(color_maps[exp_cat])): 
                df = exp.load_training_stats()
                ax = sns.lineplot(x=xaxis, y=yaxis, data=df, color=color)
                names.append(f'{exp_cat}/'+ exp.name)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        plt.legend(names)
        plt.title(f'{yaxis} on {self.dataset_name} during Labeling')

    def plot_dataset_acc(self):

        fig, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=200)

        names   = []
        all_res = []

        avg = 87.5
        for exp, color in zip(self.exps['object'], cycle(sns.color_palette("Paired")[3:])): 
            res = exp.load_history_dataset_acc()  
            res['name'] = exp.name      
            acc = exp.load_training_stats()
            sns.lineplot(x='round', y='AP', data=acc, color=color, ax=axes[0])
            names.append(exp.name)     
            all_res.append(res)

        axes[0].legend(names, loc='lower right', bbox_to_anchor=(1, 0.355))
        axes[0].set_xlabel('')
        axes[0].set_xticklabels([])
        axes[0].set_ylabel('AP@IOU[0.50:0.95] of Model Predictions')
        axes[0].set_title(f'Influence of Error Fixing Methods on {self.dataset_name}')

        plt.subplots_adjust(hspace=0.1)
        all_res = pd.concat(all_res)

        sns.barplot(x='round', y='AP', hue='name', data=all_res, ax=axes[1],
                palette= sns.color_palette("Paired")[3:])
        plt.ylabel("AP@IOU[0.50:0.95] on Validation Set")
        plt.xlabel("Labeling Rounds")
        plt.yticks(list(range(10,110,10)))
        plt.axhline(y=avg, linestyle ="--", linewidth=0.75)
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0.355))

class Visualizer:

    def __init__(self, font_path, 
                       save_base_path,
                       figure_size = (10, 5),
                       figure_dpi  = 200):

        """Handling the plotting configurations 
        """

        self.font_path   = os.path.abspath(font_path)
        self.figure_size = figure_size
        self.figure_dpi  = figure_dpi
        self.save_base_path = save_base_path

    def initialize_mpl_fonts(self):

        from matplotlib.font_manager import _rebuild; _rebuild()

        fp = FontProperties(fname=self.font_path)

        rcParams['font.serif']  = fp.get_name()
        rcParams['font.family'] = 'serif'

    def create_customized_plot(self, plot_function, save_name=None):

        with plt.style.context(['scatter', 'no-latex']):
            
            self.initialize_mpl_fonts()

            plot_function()

            if save_name is not None:
                os.makedirs(self.save_base_path, exist_ok=True)
                plt.savefig(f'{self.save_base_path}/save_name.png')
            else:
                plt.show()

    def create_simple_plot(self, plot_function, save_name=None):

        def wrapped_plot_function():
            plt.figure(figsize=self.figure_size, dpi=self.figure_dpi)
            plot_function()

        self.create_customized_plot(wrapped_plot_function, save_name)