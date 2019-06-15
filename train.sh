echo "Training fold 0 ..."
python working/keras_retinanet/bin/train.py --weights working/snapshots/resnet50_coco_best_v2.1.0.h5 --batch-size 1 --epochs 20 --steps 8000 --snapshot-path working/snapshots/fold0/ --image-min-side 512 --image-max-side 512 csv input/annotations/fold0_train_annotations.csv input/classes.csv --val-annotations input/annotations/fold0_val_annotations.csv

echo "Training fold 1 ..."
python working/keras_retinanet/bin/train.py --backbone resnet101 --batch-size 1 --epochs 20 --steps 8000 --snapshot-path working/snapshots/fold1/ --image-min-side 512 --image-max-side 512 csv input/annotations/fold1_train_annotations.csv input/classes.csv --val-annotations input/annotations/fold1_val_annotations.csv

echo "Training fold 2 ..."
python working/keras_retinanet/bin/train.py --weights working/snapshots/resnet50_coco_best_v2.1.0.h5 --batch-size 1 --epochs 20 --steps 8000 --snapshot-path working/snapshots/fold2/ --image-min-side 512 --image-max-side 512 --random-transform csv input/annotations/fold2_train_annotations.csv input/classes.csv --val-annotations input/annotations/fold2_val_annotations.csv

echo "Training fold 3 ..."
python working/keras_retinanet/bin/train.py --batch-size 1 --epochs 20 --steps 8000 --snapshot-path working/snapshots/fold3/ --image-min-side 512 --image-max-side 512 csv input/annotations/fold3_train_annotations.csv input/classes.csv --val-annotations input/annotations/fold3_val_annotations.csv