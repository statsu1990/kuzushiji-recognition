# kuzushiji-recognition
The 15th place solution in the Kaggle Kuzushiji Recognition competition<br>
https://www.kaggle.com/c/kuzushiji-recognition<br>

The outline of my solution is as follows.
- Detect with Centernet (HourglassNet backbone)
- Classify character classes with Resnet base model
The final private leaderboard score was 0.900.

![mrc](https://github.com/statsu1990/kuzushiji-recognition/blob/master/recognition_flow.png)<br>

# Image preprocessing
- to gray scale
- gaussian filter
- gamma correction
- ben's preprocessing


# Detection
### Inference
Use the two-stage Centernet to detect the bounding box of the character by the following procedure.
- Step 1: Resize the image to 512x512 and estimate bounding box 1 with the Centernet1.
- Step 2: Use the bounding box 1 to remove the outside of the outermost bounding box in the image.
- Step 3: Resize the image to 512x512 and estimate bounding box 2 with the Centernet2.
- Step 4: Ensemble bounding boxes 1 and 2 to create the final bounding box.

### Model Architecture
- Centernet1 is an ensemble of two Centernets (based on one stack hourglassnet).
- Centernet2 is an ensemble of two Centernets (based on one stack hourglassnet).

### Training
About centernet1, it is as follows.
- Training data: Use 80% of all data. (Create two models by changing the data division with random numbers.)
- Data augmentation: horizontal movement, brightness adjustment
Data expansion was essential to prevent overlearning.

About centernet2 is as follows.
- Training data: Use 80% of all data. (Create two models by changing the data division with random numbers)
- Data augmentation: Random erasing, horizontal movement, brightness adjustment
The effect of horizontal data augmentation was weak because the input image was removed outside of the bounding box. Therefore, Random erasing was indispensable.


# Classification
### Inference
Use the following procedure to classify character labels using three ensemble models of Resnet base.
- Step 1: Crop text image from original image using estimated bounding box and resize to 64x64.
- Step 2: Classify text labels with 3 Resnet base models using test time augmentation (9 types of horizontal movement).
- Step 3: Ensemble the classification results of the three models and estimate the final classification results.

### Model Architecture
- Resnet base1: Log(bounding box aspect ratio) is concatenated at FC layer.
- Resnet base2: Changed Training data from Resnet base1.
- Resnet base3: The architecture is the same as Resnet base1. A pseudo-labeled input from the above-mentioned Detection model, Resnet base1 and 2 ensemble models was added to training data. 

### Training
Each model is the same except that learning data is changed as described above and pseudo-labeling is used.
- Learning data: Use 80% of all data.
- Data expansion: horizontal movement, rotation, zoom, Random erasing


# Hardware
All models were trained using one GTX 1080 on my home server.
