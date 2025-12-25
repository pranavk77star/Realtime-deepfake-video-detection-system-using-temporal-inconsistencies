# train_with_val.py
import os
import math
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="split_dataset", help="dataset root (train/val/test)")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs_head", type=int, default=12)
parser.add_argument("--epochs_fine", type=int, default=10)
parser.add_argument("--out", type=str, default="tmp_checkpoint/finetuned.keras")
args = parser.parse_args()

DATA_DIR = args.data_dir
IMG_SIZE = args.img_size
BATCH = args.batch
EPOCHS_HEAD = args.epochs_head
EPOCHS_FINE = args.epochs_fine
OUT_PATH = args.out
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Using TF", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

# --- augmentation & generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.05,
    zoom_range=0.08,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    fill_mode="reflect"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

# compute class weights
from sklearn.utils import class_weight
y_train = train_gen.classes
classes = np.unique(y_train)
cw = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {i: float(cw[idx]) for idx,i in enumerate(classes)}
print("Class weights:", class_weights)

# --- build model (EfficientNetB0 backbone) ---
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# freeze base initially
base.trainable = False

# small head with regularization
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.Dropout(0.25)(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# --- callbacks ---
ckpt = callbacks.ModelCheckpoint(OUT_PATH, monitor="val_loss", save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)

# --- train head ---
steps_train = math.ceil(train_gen.samples / BATCH)
steps_val = math.ceil(val_gen.samples / BATCH)

print("Training head for", EPOCHS_HEAD, "epochs")
history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[ckpt, es, rlr]
)

# --- unfreeze for fine-tuning ---
print("Unfreezing top layers of base for fine-tune")
base.trainable = True

# optionally freeze lower layers, unfreeze last N layers:
fine_tune_at = len(base.layers) - 30
for i, layer in enumerate(base.layers):
    layer.trainable = True if i >= fine_tune_at else False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning for", EPOCHS_FINE, "epochs")
history2 = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=EPOCHS_FINE,
    initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0,
    class_weight=class_weights,
    callbacks=[ckpt, es, rlr]
)

print("Training finished. Best model saved to:", OUT_PATH)
