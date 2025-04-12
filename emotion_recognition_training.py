import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential

#  Step 1: Define dataset path
BASE_PATH = "C:/Users/common-research/Desktop/dataset"

train_path = os.path.join(BASE_PATH, "train")
test_path = os.path.join(BASE_PATH, "test")

#  Step 2: Count number of images per class
def count_classes(path, set_name):
    class_counts = {}
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    df = pd.DataFrame(class_counts, index=[set_name])
    return df

train_counts = count_classes(train_path, 'Train')
test_counts = count_classes(test_path, 'Test')

print("\nTraining Data Distribution:\n", train_counts)
print("\nTesting Data Distribution:\n", test_counts)

#  Step 3: Visualize sample images
def show_sample_images(directory, title):
    plt.figure(figsize=(14, 4))
    i = 1
    for label in os.listdir(directory):
        image_path = os.path.join(directory, label, os.listdir(os.path.join(directory, label))[0])
        img = tf.keras.utils.load_img(image_path)
        plt.subplot(1, 7, i)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        i += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_sample_images(train_path, "Sample Training Images")
show_sample_images(test_path, "Sample Testing Images")

#  Step 4: Prepare ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    batch_size=256,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    batch_size=256,
    color_mode='grayscale',
    class_mode='categorical'
)

#  Step 5: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    Conv2D(256, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

#  Step 6: Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  Step 7: Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)

#  Step 8: Save the model
model.save("emotion_model2.h5")
print("Model saved as emotion_model2.h5")

#  Step 9: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()