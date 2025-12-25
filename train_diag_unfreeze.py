# train_diag_unfreeze.py  (diagnostic)
# minimal edits to allow everything to train quickly for tiny dataset
import os, math, argparse, random, shutil
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.backend.set_image_data_format("channels_last")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="tmp_small")   # use the tiny set
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--out", type=str, default="tmp_checkpoint/diag_unfreeze.keras")
args = parser.parse_args()

DATA_DIR=args.data_dir; IMG_SIZE=args.img_size; BATCH=args.batch

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(os.path.join(DATA_DIR,"train"),
    target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, class_mode="binary", shuffle=True, color_mode="rgb")
val_gen = val_datagen.flow_from_directory(os.path.join(DATA_DIR,"val"),
    target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, class_mode="binary", shuffle=False, color_mode="rgb")

from tensorflow.keras.applications import EfficientNetB0
input_shape = (IMG_SIZE,IMG_SIZE,3)
print("Building EfficientNetB0; trying to load imagenet weights (might fail)...")
try:
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    print("Loaded ImageNet weights.")
except Exception as e:
    print("Could not load ImageNet weights (ok for now):", e)
    base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)

# ===== DIAGNOSTIC change: make base trainable (so whole network can learn) =====
base.trainable = True

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)   # smaller head for quick test
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs=base.input, outputs=out)

# higher LR so learning is fast on tiny set
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

ck = callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

steps_train = max(1, math.ceil(train_gen.samples / BATCH))
steps_val   = max(1, math.ceil(val_gen.samples / BATCH))

model.fit(train_gen, steps_per_epoch=steps_train, validation_data=val_gen,
          validation_steps=steps_val, epochs=args.epochs, callbacks=[ck,es])
print("DONE")
