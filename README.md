# Grab AI for SEA

This project is an implementation of [keras-retinanet by Fizyr](https://github.com/fizyr/keras-retinanet) for the Grab AI for SEA computer vision challenge.

## Installation 

I used an AWS EC2 instance running Python 3.6.7 for the project, all requirements are detailed in `requirements.txt`.

`pip install -r requirements.txt`

The top-level directory will be referred to as `$TOP`.

### Keras-RetinaNet

To install keras-retinanet:

`cd $TOP/working ; pip install .` 

## Data

Due to file size limitations, the [train](http://imagenet.stanford.edu/internal/car196/cars_train.tgz) and [test](http://imagenet.stanford.edu/internal/car196/cars_test.tgz) images are not provided in the repository. 

In `$TOP`, run the following commands: 

```
wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
tar zxvf cars_train.tgz
tar zxvf cars_test.tgz
mv cars_train input/cars_train
mv cars_test input/cars_test 
```

To prepare the data for training, run the following script to get all the data into the necessary formats.

`sh prepare_data.sh`

## Training 

Training is done through stratified 4-fold cross validation by class with the final prediction done through an ensemble of the 4 models.

Each model is trained with different parameters such as intialisation weights, data augmentation and model backbone.

To train all models in the ensemble:

`sh train.sh`

Comment out lines if you only wish to train specific models. 

## Trained Models 

The training code for the models depends on the existence of the pretrained models.

To download the trained models as well as the pretrained COCO model I used for the challenge, in `$TOP/working` use: 

```
wget --no-check-certificate \
                         -r \
'https://docs.google.com/uc?export=download&id=1suNmbfHutOdw5X_xx5ZUPxxQhH7hJTU3' \
-O snapshots.tar.gz
```

Extract the folder into `working/snapshots`.

## Inference 

`sh predict.sh` 

This should produce predictions for the test data in `$TOP/FinalSubmission.csv` in the format:

`[imageName, box_x, box_y, box_w, box_h, class, className, confidence]`

A text file will also be produced in `$TOP/FinalSubmission.txt` for submission on the [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) evaluation server.

Due to time and cost I only trained each model for 20 epochs and the ensemble model achieves 59.20% accuracy on the test set. I believe with more training the accuracy and mAP of the ensemble model will improve.

## Acknowledgements

I used code from the following repositories: 

https://github.com/fizyr/keras-retinanet

https://github.com/ahrnbom/ensemble-objdet