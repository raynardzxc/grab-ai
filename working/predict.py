from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet.models import load_model
from ensemble_objdet.ensemble import GeneralEnsemble
import numpy as np
import argparse
import glob
import os

MODEL0_PATH = 'working/snapshots/fold0/mAP45.h5'
MODEL1_PATH = 'working/snapshots/fold1/mAP34.h5'
MODEL2_PATH = 'working/snapshots/fold2/mAP42.h5'
MODEL3_PATH = 'working/snapshots/fold3/mAP42.h5'
OUTPUT_DIR = '.'
LABEL_PATH = 'input/classes.csv'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to directory containing input images")
ap.add_argument("-o", "--output", default=OUTPUT_DIR,
    help="path to directory to store predictions")
args = vars(ap.parse_args())

model0 = load_model(MODEL0_PATH, backbone_name="resnet50", convert=True)
model1 = load_model(MODEL1_PATH, backbone_name="resnet101", convert=True)
model2 = load_model(MODEL2_PATH, backbone_name="resnet50", convert=True) 
model3 = load_model(MODEL3_PATH, backbone_name="resnet50", convert=True)

model_list = [model0, model1, model2, model3]

# load the class label mappings
LABELS = open(LABEL_PATH).read().strip().split('\n')
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

imagePaths = glob.glob(os.path.join(args["input"], "*"))

test_predictions = []
test_pids = []

# loop over the input image paths
for (i, imagePath) in enumerate(imagePaths):

    print ("[INFO] predicting on image {} of {}".format(i+1, len(imagePaths)))

    individual_preds = []

    image = read_image_bgr(imagePath)
    image = preprocess_image(image)
    (image, scale) = resize_image(image, min_side=512, max_side=512)
    image = np.expand_dims(image, axis=0)

    for index, model in enumerate(model_list):
        prediction = model.predict_on_batch(image)
        bboxes = prediction[0][0][:1] / scale
        scores = prediction[1][0][:1]; scores = np.clip(scores, 0, 1)
        labels = prediction[2][0][:1]
        bboxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in bboxes]
        final_pred = [list(_) + [labels[i], scores[i]] for i, _ in enumerate(bboxes)]
        individual_preds.append(final_pred)

    test_predictions.append(individual_preds)
    test_pids.append(imagePath.split("/")[-1].split(".")[0])

import pandas as pd 

# parse the predictions
det_df = pd.DataFrame() 
for picIndex, eachDet in enumerate(test_predictions): 
    for modelIndex, eachModelPred in enumerate(eachDet): 
        tmp_df = pd.DataFrame(np.asarray(eachModelPred)) 
        tmp_df.columns = ["x", "y", "w", "h", "CLASS", "rawScore"]
        tmp_df["pid"] = test_pids[picIndex]
        tmp_df["modelId"] = modelIndex
        det_df = det_df.append(tmp_df)

detections_list = [] 
for each_pid in test_pids: 
    tmp_df = det_df[det_df.pid == each_pid] 
    individual_pid_dets = []
    for each_model in range(1): 
        individual_model_dets = [] 
        tmp_model_df = tmp_df[tmp_df.modelId == each_model] 
        for rowNum, row in tmp_model_df.iterrows(): 
            individual_model_dets.append([row.x, row.y, row.w, row.h, row.CLASS, row.rawScore])
        individual_pid_dets.append(individual_model_dets) 
    detections_list.append(individual_pid_dets)

# ensemble predictions
ensemble_dets = [] 
for each_det in detections_list: 
    ensemble_dets.append(GeneralEnsemble(each_det))

# assemble dataFrame
import pandas as pd 
df = pd.DataFrame() 
for index, each_det in enumerate(ensemble_dets): 
    tmp_df = pd.DataFrame({"pictureId": test_pids[index],
                           "x": [box[0] for box in each_det],
                           "y": [box[1] for box in each_det],
                           "w": [box[2] for box in each_det], 
                           "h": [box[3] for box in each_det],
                           "classId": [box[4] for box in each_det],
                           "score": [box[5] for box in each_det]})
    df = df.append(tmp_df) 

# clean up
df['classId'] = df.classId.astype('int')
df = df.reset_index(drop=True)
df_txt = df.copy()

# Add className
labels_df = pd.Series([]) 

for i in range(len(df)): 
    val = df.iloc[i]['classId']
    labels_df[i] = LABELS[val] 

df.insert(6, "className", labels_df) 
df.to_csv(os.path.join(args["output"], "FinalSubmission.csv"), index=False)
print ("[INFO] Predictions for csv generated and saved in {}".format(os.path.join(args["output"], "FinalSubmission.csv")))

df_txt = df_txt.sort_values(by=['pictureId']) 
columnsTitles = ['classId']
df_txt = df_txt.reindex(columns=columnsTitles)

## evaluation classes start from 1
df_txt = df_txt + 1
df_txt.to_csv(os.path.join(args["output"], "FinalSubmission.txt"), header=False, index=False, sep=' ')
print ("[INFO] Predictions for evaluation generated and saved in {}".format(os.path.join(args["output"], "FinalSubmission.txt")))