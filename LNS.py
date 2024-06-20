pip install datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from datasets import load_dataset

# Load the LyNoS dataset
dataset = load_dataset("andreped/LyNoS")

# Define the input shape
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_DEPTH = 128
IMG_CHANNELS = 1

# Define the data loader function
def load_data(dataset, batch_size):
    """
    Loads and preprocesses the data for training or testing.
    """
    data = []
    labels = []

    for d in dataset:
        # Load the CT scan and corresponding label
        ct_scan = d["ct"]
        label = d["lymphnodes"]
        
        # Preprocess the label
        label_data = label_data / label_data.max()
        label_data = label_data.astype(np.float32)

        # Preprocess the CT scan
        ct_scan = ct_scan / ct_scan.max()
        ct_scan = ct_scan.astype(np.float32)

        # Preprocess the label
        label = label / label.max()
        label = label.astype(np.float32)

        # Append to the data and labels lists
        data.append(ct_scan)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    data_ds = tf.data.Dataset.from_tensor_slices((data, labels))
    data_ds = data_ds.batch(batch_size)

    return data_ds

# Load the training and testing data

train, test = train_test_split( dataset["test"], test_size=0.2, random_state=0)

train_data = load_data(train, batch_size=4)
test_data = load_data(test, batch_size=4)

# Define the U-Net architecture
def unet_model(input_shape, num_classes):
    """
    Defines the U-Net architecture for image segmentation.
    """
    inputs = tf.keras.Input(input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(conv1)

    conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(conv2)

    conv3 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(conv3)

    conv4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling3D((2, 2, 2), padding='same')(conv4)

    # Bottleneck
    conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    up6 = tf.keras.layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv4])
    conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv3])
    conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2])
    conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1])
    conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    conv10 = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    return model

# Compile the model
model = unet_model((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS), 2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=100, validation_data=test_data)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data)
print(f'Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}')
