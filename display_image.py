import numpy as np
import matplotlib.pyplot as plt
import csv

# Load CSV file
csv_file = 'train.csv'  # Replace with your CSV file path

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    data = next(reader)  # Read the first row of actual image data
    data = list(map(float, data))  # Convert data to floats

# Ignore the first value and reshape the rest to 28x28
image_data = np.array(data[1:])  # Ignore the first value
image_data = image_data.reshape((28, 28))

# Display the image
plt.imshow(image_data, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()
