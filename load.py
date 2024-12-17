import tensorflow_datasets as tfds

# Load the tf_flowers dataset
dataset, dataset_info = tfds.load("tf_flowers", as_supervised=True, with_info=True)

# Print dataset information
print(dataset_info)

# Split the dataset into training and testing
train_data = dataset['train'].take(3500)  # Use 3500 images for training
test_data = dataset['train'].skip(3500)  # Use remaining images for testing

# Save train and test data for other scripts (optional)
# Example: Save dataset splits to TFRecord or pickle if needed
