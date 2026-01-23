import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = "data/image_dataset/train"
VAL_DIR = "data/image_dataset/val"

# ======================
# DATA GENERATORS
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Class indices:", train_ds.class_indices)

# ======================
# CLASS WEIGHTS
# ======================
pcos_count = train_ds.classes.sum()
non_pcos_count = len(train_ds.classes) - pcos_count

class_weight = {
    0: pcos_count / non_pcos_count,
    1: 1.0
}

print("Class weights:", class_weight)

# ======================
# MODEL
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ======================
# CALLBACKS
# ======================
checkpoint = ModelCheckpoint(
    "models/image_cnn_classifier.h5",
    monitor="val_auc",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=4,
    restore_best_weights=True
)

# ======================
# TRAIN
# ======================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop]
)

print("\nTraining completed. Best model saved.")
