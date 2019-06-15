import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

cwd = os.getcwd()
devkit_path = os.path.join(cwd,'input/devkit')
train_path = os.path.join(cwd,'input/cars_train')

cars_meta = loadmat(devkit_path + '/cars_meta.mat')
cars_train_annos = loadmat(devkit_path + '/cars_train_annos.mat')

labels = [c for c in cars_meta['class_names'][0]]
labels = pd.DataFrame(labels, columns=['labels'])

frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_train = pd.DataFrame(frame, columns=columns)
df_train['class'] = df_train['class']-1 
df_train['fname'] = ['../../input/cars_train/' + f for f in df_train['fname']]

df_train = df_train.merge(labels, left_on='class', right_index=True)
df_train = df_train.sort_index()

columnsTitles = ['fname','bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2','labels','class']
df_train=df_train.reindex(columns=columnsTitles)
df_classes = df_train.copy()

df_classes = df_classes.drop(columns=['fname','bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
df_classes = df_classes.drop_duplicates(subset=['class'])
df_classes = df_classes.sort_values(by=['class'])

classes = ['labels','class']
df_train.to_csv('input/train_labels.csv',index = False)
df_classes.to_csv('input/classes.csv', header=False, columns = classes, index = False)

from sklearn.model_selection import StratifiedKFold

train_path = os.path.join(cwd,'input/train_labels.csv')
train = pd.read_csv(train_path)
target = train['class'].copy()

def assign_folds(orig_df, num_folds, seed=2019):
    # Stratified splits
    np.random.seed(seed) 
    df = orig_df.copy() 
    df["fold"] = None  
    skf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True) 
    fold_counter = 0 
    for train_index, test_index in skf.split(target, target):
        df["fold"].iloc[test_index] = fold_counter
        fold_counter += 1 
    return df

folds_df = assign_folds(train, 4)
folds_df.to_csv('input/fold_train_labels.csv',index = False)

labels_df = pd.read_csv(os.path.join(cwd, 'input/fold_train_labels.csv'))
save_annotations_dir = os.path.join(cwd, "input/annotations/")

def generate_annotations(fold, labels_df=labels_df, save_annotations_dir=save_annotations_dir):
    df_train = labels_df[labels_df['fold'] != fold] 
    df_val = labels_df[labels_df['fold'] == fold] 
    df_train = df_train[['fname','bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2','labels']]
    df_val = df_val[['fname','bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2','labels']]
    print(df_train.shape, df_val.shape)
    if not os.path.exists(save_annotations_dir): os.makedirs(save_annotations_dir)
    df_train.to_csv(os.path.join(save_annotations_dir, "fold{}_train_annotations.csv".format(fold)), index=False, header=False)
    df_val.to_csv(os.path.join(save_annotations_dir, "fold{}_val_annotations.csv".format(fold)), index=False, header=False)

generate_annotations(0)
generate_annotations(1)
generate_annotations(2)
generate_annotations(3)