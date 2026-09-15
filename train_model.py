import os

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import InceptionV3

from tensorflow.keras import layers, models, regularizers

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

def cosine_annealing(epoch):

initial_lr = 1e-4

epochs = 25

return initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2

--- Paths ---

IMG_SIZE = (299, 299)

BATCH_SIZE = 16

base_dir = 'chestx_ray\Data'

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')

--- Augmentation ---

train_datagen = ImageDataGenerator(

rescale=1./255,

rotation_range=20,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.15,

zoom_range=0.2,

brightness_range=(0.7, 1.3),

horizontal_flip=True,

fill_mode='nearest'

)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(

train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True)

val_data = val_test_datagen.flow_from_directory(

val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

test_data = val_test_datagen.flow_from_directory(

test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

--- Build Model ---

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))

base_model.trainable = False

model = models.Sequential([

base_model,

layers.GlobalAveragePooling2D(),

layers.BatchNormalization(),

layers.Dropout(0.5),

layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))

])

model.compile(

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),

metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

)

early_stopping = EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(patience=3, factor=0.3, monitor='val_loss')

checkpoint = ModelCheckpoint('best_inceptionv3_model.keras', save_best_only=True, monitor='val_loss')

lr_scheduler = LearningRateScheduler(cosine_annealing)

--- Phase 1: Feature Extraction (frozen base) ---

model.fit(

train_data,

validation_data=val_data,

epochs=15,

callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]

)

--- Phase 2: Deep Fine-Tuning (unfreeze much of base) ---

base_model.trainable = True

for layer in base_model.layers[:-100]:

layer.trainable = False

model.compile(

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),

loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),

metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

)

model.fit(

train_data,

validation_data=val_data,

epochs=25,

callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler]

)

--- Evaluate on Test Set ---

model = tf.keras.models.load_model('best_inceptionv3_model.keras')

loss, accuracy, auc, precision, recall = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

--- Confusion Matrix & Report ---

y_pred_probs = model.predict(test_data)

y_pred = (y_pred_probs > 0.5).astype(int).flatten()

y_true = test_data.classes

labels = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.show()

print(classification_report(y_true, y_pred, target_names=labels))

this is the model code that we trained