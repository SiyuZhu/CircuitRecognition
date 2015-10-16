Digital Gate Recognition CNN
============================
Uses caffe to train a convolutional neural network (CNN) that can categorize
digital gates. The current implementation can categorize AND, NAND, NOT, OR, 
NOR, and XOR  gates. Run all code while in this directory.

```sh
unzip train_val_data.zip
cd tools
make all
cd ..
./preprocess_images.sh
./create_imagenet_data.sh
./train_lenet.sh
# Ctrl-C to stop the training after first snapshot. classifier was
# programmed to use the first snapshot, but this can easily be changed.
# see deploy_classifier.py
python deploy_classifier.py
```