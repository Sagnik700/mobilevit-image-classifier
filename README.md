# CSP Lab Mobilevit Image Classifier


This project is developed under the Computer Security and Privacy Lab of University of Goettingen. Images inside the public and private folders in assets will be classified as either Sensitive(Private) or Non-sensitive(Public).

*****
## Project Structure:-

Assests folder(/app/src/main/assets/):-
1. mobilevit_xxs_2.tflite - Tflite mobile version of Mobilevit For Image Privacy Classification
2. vit_mobile1.ptl - Pytorch mobile version of ViTForImageClassification
3. flickr/private - Folder containing private images
4. flickr/public - Folder containing public images
*****

*****
## How to run the Android project:-

1. Clone the repository into local.
2. Import the project in Android Studio.
3. Perform a clean Gradle build in Android Studio(if any complications arise, delete the '.idea' and '.gradle' folder from the project structure and perform a clean build).
4. Run the application in an emulator or Android mobile.
5. Click on Process button.
6. The image privacy classification of either Sensitive or Non-sensitive will be displayed in the Log space of the Android Studio.
*****


*****
## Jupiter notebook list (inside 'jyputer notebook' directory):-

1. Fine-tuning the Vision Transformer.ipynb - Fine-tunes the pretrained ViTForImageClassification from Pytorch Hugging Face library with provided dataset(Flickr imageset). The dataset that is provided is being divided into training, validation and test sets to first train the last classification layer upon our new dataset and then to validate and test its predictions. The notebook is very well documented inside and thus every information required can be found while running it. At the end of the notebook a mobile version of the model is generated stored with the .ptl extension.

2. Mobilevit_final.ipynb - Trains a MobileVit model from scratch (Architechture defined inside) with the provided dataset. The notebook is very well documented inside and thus every information required can be found while running it. At the end of the notebook a mobile version of the model is generated stored with the .tflite extension.
*****

*****
## Limitations, improvements and important notes:-

1. A threshold is selected for classification of the image privacy class in the Android application, and it is having a value of 0.2. Multiple values can be tried and tested with the a dummy test set to get the best value for which accuracy is most.
2. Mobilevit was found to be very sensitive to imbalanced dataset to a 1:1 ratio for the 2 classes of images were used for training. Since the maximum number of private images available were close to 6000, an equal number of public images were taken for training. More private images could have given us better training because the number of public images is close to 22000 and the more data we would have, better the model will perform.
3. Hyperparameter tuning (adjusting the learning rate, epochs, batch size, ADAM properties, etc.) can be a good idea during the training process of the model in the jupyter notebook.
4. vit_mobile1.ptl - Pytorch mobile version of ViTForImageClassification has been kept in the assets folder and the notebook that generated it is also uploaded. While testing out the model in Android, it gave very bad accuracy for Private images. Almost all of the images were getting classified as public and that can be due to improper training in the notebook or forwarding of wrongly parsed image data to the pytorch mobile model in Android. This issue needs to be debugged thoroughly for a better Root Cause Analysis.
5. Any kind of debugging on these mobile version of models require a very close analysis with their jupyter notebook counterpart to understand the root of any error or anomaly. For example, the parsed images in the train/validation set in the notebook can be printed and same can be done for the parsed images in Android to see if the tokenization process of the images are same on both ends or not. If yes, then the parsed image array(in this case a multi dimensional Tensor) can be exported and fed to the mobile version of the model in the Android application to see whether it is classifying same as of the notebook model.
