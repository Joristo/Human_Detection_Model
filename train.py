# Import necessary libraries
from human_detection_model import train_model, load_and_process_images, save_and_load_np_array, plot_training_history

# Load and process images for the 'person' category from the 'data' folder
person_images = load_and_process_images('data')

# Save the processed images and load them back using a numpy array
save_and_load_np_array(person_images, 'processed_data', 'person_images.npy')

# Train a model using the processed data in the 'processed_data' folder
model, history = train_model('processed_data')

# Plot the training history to visualize model performance during training
plot_training_history(history)

# Save the trained model to a file named 'my_model.h5'
model.save('my_model.h5')