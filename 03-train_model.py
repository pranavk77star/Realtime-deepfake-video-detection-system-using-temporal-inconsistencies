# 03-train_model.py  (Final Stable Version)
import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.activations import swish
import matplotlib.pyplot as plt

# =========================
# Setup
# =========================
print('TensorFlow version:', tf.__version__)

if len(sys.argv) < 3:
    print("Usage: python 03-train_model.py <split_dataset_subfolder> <output_model_name>")
    print("Example: python 03-train_model.py .\\split_dataset\\Celeb celeb_model.keras")
    sys.exit(1)

dataset_path = sys.argv[1]      # e.g., ./split_dataset/Celeb
output_model_name = sys.argv[2] # e.g., celeb_model.keras

checkpoint_dir = './tmp_checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# =========================
# Parameters
# =========================
input_size = 128
batch_size_num = 32
num_epochs = 20  # you can increase to 25 later

train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# =========================
# Data Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# =========================
# Model
# =========================
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

model = Sequential([
    efficient_net,
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

model.summary()

# =========================
# Compile
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Callbacks
# =========================
checkpoint_path = os.path.join(checkpoint_dir, output_model_name)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# =========================
# Train
# =========================
print("\n✅ Starting training...")
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=callbacks
)
print("\n✅ Training complete.")

# =========================
# Plot Results
# =========================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'bo-', label='Train Acc')
plt.plot(epochs, val_acc, 'b-', label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'ro-', label='Train Loss')
plt.plot(epochs, val_loss, 'r-', label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# =========================
# Evaluate Best Model
# =========================
print("\n🔍 Loading best model for testing...")
best_model = load_model(checkpoint_path, custom_objects={"swish": swish})

test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)
print("✅ Prediction done!")

# =========================
# Save Results
# =========================
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
test_results.to_csv(f"test_predictions_{os.path.basename(dataset_path)}.csv", index=False)

print(f"✅ Predictions saved to test_predictions_{os.path.basename(dataset_path)}.csv")
print("✅ Best model saved at:", checkpoint_path)

