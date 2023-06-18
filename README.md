# Haar Cascade Classifiers

Haar Cascade is a machine learning object detection algorithm where a cascade function is trained from a lot of positive and negative images (where positive images are those where the object to be detected is present, negative are those where it is not).

For this code implementation, the pre-trained Haar cascade algorithms available with OpenCV is used.

Those can be found here : https://github.com/opencv/opencv/blob/master/data/haarcascades



# Tuining the classifier

### **detectMultiSale(input image, Scale, Neighbours)**


**Scale** : Specifies how much we reduce the image size each time we scale. E.g. in face detection we typically use 1.3. This means we reduce the image by 30% each time it's scaled. Smaller values will take longer to compute, but will increase the rate of detection

**Neighbours** : Specifies the number of neighbours each potential window should have in order to consider it a positive detection. Typically set between 3-6. It acts as sensitivity setting, low values will sometimes detect multiple faces over a single face. High values will ensure less false positives, but you may miss some faces.

# Running the code
This code uses the webcam to get the live feed and detects the face, smile and eyes.

To run the code included in this repository:

1> Install the OpenCV and python

2> Clone the repository

3> run **python3 detect.py**


