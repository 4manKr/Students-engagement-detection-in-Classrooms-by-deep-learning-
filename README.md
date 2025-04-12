# Students-engagement-detection-in-Classrooms-by-deep-learning-
The complete entire process  can be divided into structured one: Dataset Description, Data Preprocessing, CNN Model  Architecture, and Model Training and Evaluation. 

![image](https://github.com/user-attachments/assets/221a169c-3c67-4a22-a987-26b00c969d5f)

This is what the study procedure for student engagement detection via deep learning 
techniques looks like: emotion recognition from facial expressions. This is what is executed 
using convolutional neural networks (CNNs) constructed with Python and Keras frameworks, 
as shown in the attached emotion_recognition_training.py script. The complete entire process 
can be divided into structured one: Dataset Description, Data Preprocessing, CNN Model 
Architecture, and Model Training and Evaluation. The methodology attempts to catch faded 
emotional cues through subtle facial expressions exhibited during classrooms' interaction and 
links them with engagement levels. 

**Dataset Description (FER-2013)**********

**Training Data**
![image](https://github.com/user-attachments/assets/a6bade53-f976-466b-80b7-1a59571f03a1)

**Testing Data**
![image](https://github.com/user-attachments/assets/0e42fa7d-c9bc-4d49-bf56-75037d92f93c)

Image Data Augmentation
This is perhaps the main feature for enhancement in your implementation by using data augmentation via Keras ImageDataGenerator class: 
o	Horizontal Flipping: Simulates turning the head toward left or right. Zooming: Slight zoom mimics distance from camera.

o	Shifting: Mimics repositioning of face.

o	Rotation: Accounts for head tilt. 


![image](https://github.com/user-attachments/assets/46cb8f70-f34c-47af-b4c5-6b329006c4ab)

visualize how augmentations such as flip, zoom, shift, and rotation would change a single face image

**CNN Model Architecture**
The Convolutional Neural Network (CNN) at the very heart of this methodology is entirely responsible for the learning of facial emotion patterns. The model is essentially a deep CNN built with the Keras Sequential API and made up of several different layers dedicated to spatial feature learning and classification.

•	Input Layer (48x48x1 grayscale images)
Accepting a normalized image tensor in which "48" is the width and height of the image, while "1" denotes the grayscale channel.

•	Convolutional Layers (Image Feature Extractors):
In the earliest layers, sets of 3x3 filters apply across the image to extract facial features like edges, textures, and contours. Each successive convolutional layer learns higher-level abstractions-edged to facial parts, and ultimately to entire expressions.

•	ReLU Activation Function
The Rectified Linear Unit (ReLU), which brings non-linearity after every convolutional layer, permits the network to learn complex decision boundaries. If there is no ReLU, the network can learn only linear mappings.

•	Batch Normalization
Batch Normalization is the term used to explain the normalization of input to each layer. Reduces Internal Covariate Shift thus speeding up training with higher learning rates and, on occasion, acts as a regularizer for dropout.

•	Max Pooling Layers
Max pooling is the process of down-sampling feature maps by taking the maximum value from the 2x2 region. This reduces both the dimensionality and computational load on the system, while at the same time enhancing some degree of robustness to small changes in the input.

•	Dropout Layers (Regularization)
Dropout prevents the model from over-relying on any single feature, thus reducing overfitting and improving generalization by randomly switching off a certain percentage (25% to 50%) of neurons during training.

•	Flattening Layer
In this operation, the 2D matrix from the convolutional layer is converted into a 1D vector so that it can be fed into fully connected (Dense)-layers for classification.

•	Fully Connected Dense Layer (1024 units)
This layer pools all learned features and connects them to the final decision layer. Another ReLU is performed to further refine the representation before classification.

•	Output Layer (Softmax for 7 emotion classes)
The final dense layer uses a softmax function to generate a probability distribution over the seven classes. The emotion with the highest probability is selected as the prediction.

![image](https://github.com/user-attachments/assets/31fa44c9-5e1f-4e7f-8d7e-0faca767391b)



**Final Methodology**
![image](https://github.com/user-attachments/assets/a1fbd79e-7216-44bd-b4dd-1bf647662a72)


**Training Logs**

![image](https://github.com/user-attachments/assets/afee9f6e-d1a5-45fc-949d-4de8eab6c1ad)

This output in the terminal shows increasing training accuracy and decreasing training loss, thus confirming that the model is learning during the 25 epochs.

Key Observations:

•	Final Training Accuracy: 99.92%

•	Final Validation Accuracy: 60.02%

•	Final Training Loss: 0.4284

•	Final Validation Loss: 1.6816

•	Model Saved As: emotion_model2.h5


**Model Evaluation Output**

![image](https://github.com/user-attachments/assets/ba75e80a-9e84-4ac6-8968-4d0b192f8f57)

The loaded model (emotion_model2.h5) is evaluated with a test script (evaluate_model.py) that generates the output of a confusion matrix and classification report.

Confusion Matrix and Classification Report:

•	Accuracy: 99.89%

•	Precision: 99.60%

•	Recall: 99.59%

•	F1-score: 99.59%
