import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = 256  # Resize all images to 256x256
BATCH_SIZE = 32
EPOCHS = 50

# Step 1: Load and Preprocess the Dataset (tf_flowers dataset)
# Load the dataset
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)

# Check available splits
print("Dataset splits:", dataset.keys())

# In case 'test' split is not available, use 'train' and split it manually
train_data = dataset['train']

# Optionally, split the train data into a validation set manually (80-20 split)
val_size = 0.2
train_size = 1 - val_size

train_data = train_data.take(int(train_size * len(train_data)))
valid_data = dataset['train'].skip(int(train_size * len(train_data)))

# Preprocess the images (resize and normalize)
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize images
    image = tf.cast(image, tf.float32) / 255.0  # Normalize images to [0, 1]
    return image, label

train_data = train_data.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
valid_data = valid_data.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Step 2: Define the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(info.features['label'].num_classes, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Step 5: Train the Model
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data,
    callbacks=[early_stopping]
)

# Step 6: Evaluate the Model on the Validation Data
val_loss, val_acc = model.evaluate(valid_data)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

# Step 7: Optionally Save the Model
model.save('flower_classifier_model.h5')

# Optional: Plot Training History (Accuracy and Loss)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
