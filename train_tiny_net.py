# train_tiny_net.py
import numpy as np, os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train = datagen.flow_from_directory("tmp_small/train", target_size=(128,128), batch_size=8, class_mode="binary", shuffle=True)
val   = datagen.flow_from_directory("tmp_small/val",   target_size=(128,128), batch_size=8, class_mode="binary", shuffle=False)

model = models.Sequential([
    layers.Input((128,128,3)),
    layers.Conv2D(16,3,activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=80, validation_data=val)
