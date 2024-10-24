import tkinter as tk
import numpy as np
import pandas as pd
import os
import math

FILEPATH = "datasets/new_train_set.csv"

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Dataset Creator")
        self.digit = np.random.randint(0, 10)

        self.header = ["pixel" + str(i) for i in range(784)]
        self.header.insert(0, "label")
        self.header = ",".join(self.header)

        self.canvas = tk.Canvas(master, width=280, height=280, bg='black')  # Background black for MNIST-style look
        self.canvas.pack()

        self.digit_label = tk.Label(master, text="Digit: " + str(self.digit))
        self.digit_label.pack()

        self.save_button = tk.Button(master, text='Save', command=self.save_image)
        self.save_button.pack()
        
        self.clear_button = tk.Button(master, text='Clear', command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.image_data = np.zeros((28, 28), dtype=int)  # Initialize as black

    def paint(self, event):
        x, y = event.x // 10, event.y // 10
        if 0 <= x < 28 and 0 <= y < 28:
            # Draw the main white line (255)
            if self.image_data[y, x] < 255:  # Only update if it's not already white
                self.image_data[y, x] = 255
                self.update_canvas(x, y, 255)

            # Draw smooth gray surrounding pixels
            self.draw_surrounding_pixels(x, y)

    def draw_surrounding_pixels(self, x, y):
        for dx in [-2, -1, 0, 1, 2]:  # Include a larger radius for smoother transition
            for dy in [-2, -1, 0, 1, 2]:
                if (dx != 0 or dy != 0) and 0 <= x + dx < 28 and 0 <= y + dy < 28:
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance <= 2:  # Limit the distance
                        # Gradually fade to gray, with closer pixels being brighter
                        gray_value = int(255 * (1 - distance / 2))  # Full white to dark gray
                        # Update only if the new gray value is lighter (higher) than the current value
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
        self.image_data.fill(0)  # Reset to black

    def save_image(self):
        label = self.digit
        
        flattened_data = self.image_data.flatten()
        df = pd.DataFrame(flattened_data).T
        df.insert(0, 'label', label)

        file_mode = 'a' if os.path.exists(FILEPATH) else 'w'
        if not os.path.exists(FILEPATH):
            print("Hallo")
            with open(FILEPATH, 'w') as file:  # Use 'a' mode if you want to append
                file.write(self.header + "\n") 
        df.to_csv(FILEPATH, mode="a", index=False, header=False)
        
        print(label)
        self.clear_canvas()
        self.next_number()
        return df
    
    def next_number(self):
        self.digit += 1
        if self.digit > 9:
            self.digit = 0
        self.digit_label.config(text="Digit: " + str(self.digit))



if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
