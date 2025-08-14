import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

# ======================
# DATA PREPARATION
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
val_data = val_datagen.flow_from_directory(
    'dataset/val', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
test_data = test_datagen.flow_from_directory(
    'dataset/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# ======================
# MODEL CREATION
# ======================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ======================
# CALLBACKS
# ======================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

# ======================
# TRAINING PHASE 1
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ======================
# FINE-TUNE PHASE 2
# ======================
base_model.trainable = True  # unfreeze
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# ======================
# EVALUATE ON TEST DATA
# ======================
test_loss, test_acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# ======================
# SAVE MODEL
# ======================
model.save("signature_detector_v2.h5")

# ======================
# PLOT TRAINING HISTORY
# ======================
def plot_history(histories, title):
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for h in histories:
        acc += h.history['accuracy']
        val_acc += h.history['val_accuracy']
        loss += h.history['loss']
        val_loss += h.history['val_loss']

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title(f"{title} - Accuracy")

    plt.subplot(1,2,2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title(f"{title} - Loss")

    plt.show()

plot_history([history, fine_tune_history], "Signature Detector Training")
