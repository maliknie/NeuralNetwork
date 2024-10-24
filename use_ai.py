import numpy as np
import pandas as pd
import os
import pickle
import math
import tkinter as tk

from main import Network, Layer, Activation

def initialize_network(param_filename="params97_38"):
    network = Network()
    network.add(Layer(784, 512, Activation.relu))
    network.add(Layer(512, 256, Activation.relu))
    network.add(Layer(256, 128, Activation.relu))
    network.add(Layer(128, 10, Activation.softmax))
    network.load_params("params/" + param_filename + ".bin") 
    return network

class DrawingApp:
        def __init__(self, master, network):
            self.network = network
            self.master = master
            self.master.title("Predict Digit")
            self.returned_image = None
            self.prediction = None
    
            self.canvas = tk.Canvas(master, width=280, height=280, bg='black')
            self.canvas.pack()
            
            self.clear_button = tk.Button(master, text='Clear', command=self.clear_canvas)
            self.clear_button.pack()


            self.prediction_label = tk.Label(master, text="Prediction: " + str(self.prediction))
            self.prediction_label.pack()


            self.canvas.bind("<B1-Motion>", self.paint)
            self.canvas.bind("<ButtonRelease-1>", self.predict)
    
            self.image_data = np.zeros((28, 28), dtype=int)
            
    
        def paint(self, event):
            x, y = event.x // 10, event.y // 10
            if 0 <= x < 28 and 0 <= y < 28:
                if self.image_data[y, x] < 255:
                    self.image_data[y, x] = 255
                    self.update_canvas(x, y, 255)

                self.draw_surrounding_pixels(x, y)
    
        def draw_surrounding_pixels(self, x, y):
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if (dx != 0 or dy != 0) and 0 <= x + dx < 28 and 0 <= y + dy < 28:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= 2:
                            gray_value = int(255 * (1 - distance / 2))
                            if self.image_data[y + dy, x + dx] < gray_value:
                                self.image_data[y + dy, x + dx] = gray_value
                                self.update_canvas(x + dx, y + dy, gray_value)
    
        def update_canvas(self, x, y, gray_value):
            hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
            self.canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill=hex_color, outline=hex_color)
    
        def reset(self, event):
            pass
        
        def clear_canvas(self):
            self.canvas.delete("all")
            self.image_data.fill(0)
            self.prediction = None
            self.prediction_label.config(text="Prediction: " + str(self.prediction))
    
        def return_image(self):
            self.returned_image = self.image_data.reshape(1, 784) / 255.0

        def predict(self, event):
            self.return_image()
            self.prediction = self.network.guess(self.returned_image)
            self.prediction_label.config(text="Prediction: " + str(self.prediction))   

network = initialize_network()

root = tk.Tk()
app = DrawingApp(root, network)
root.mainloop()