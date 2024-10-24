import numpy as np
import matplotlib.pyplot as plt
import csv

FILEPATH = 'train.csv'

with open(FILEPATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)  
    data = next(reader) 
    data = list(map(float, data))

image_data = np.array(data[1:])
image_data = image_data.reshape((28, 28))

plt.imshow(image_data, cmap='gray')
plt.axis('off')
plt.show()
