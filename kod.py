import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# dataset path
images_folder  = "leedsbutterfly/images/"
texts_folder  = "leedsbutterfly/descriptions/"
# Load images and labels
images = []
labels = []

for category_id in range(1, 11):
    category_code = f"{category_id:03d}"

    # Load images
    category_images = glob(os.path.join(images_folder, f"{category_code}*.png"))
    for image_path in category_images:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        images.append(image)
        labels.append(category_id - 1)  # Subtract 1 because categories start from 1

# Convert data to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Create train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# Make predictions on the validation set
val_predictions = model.predict(X_val)
val_predictions_classes = np.argmax(val_predictions, axis=1)

# Display predictions and true labels with butterfly names from text files
for i in range(5):  # For the first 5 examples
    category_id = val_predictions_classes[i] + 1
    category_code = f"{category_id:03d}"
    text_file_path = os.path.join(texts_folder, f"{category_code}.txt")

    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        scientific_name = text_file.readline().strip()
        common_name = text_file.readline().strip()

    plt.imshow(X_val[i] / 255.0)  # Display the image
    plt.title(f"Prediction: {scientific_name}")
    plt.show()