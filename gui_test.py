import tkinter as tk
import numpy as np
import pandas as pd
import os

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit with Gray Edges")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.label_entry = tk.Entry(master)
        self.label_entry.pack()
        self.label_entry.insert(0, "Enter digit (0-9)")

        self.save_button = tk.Button(master, text='Save', command=self.save_image)
        self.save_button.pack()
        
        self.clear_button = tk.Button(master, text='Clear', command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.image_data = np.zeros((28, 28), dtype=int)

    def paint(self, event):
        x, y = event.x // 10, event.y // 10
        if 0 <= x < 28 and 0 <= y < 28:
            # Draw black pixel
            if self.image_data[y, x] < 255:
                self.image_data[y, x] = min(0, self.image_data[y, x] + 51)  # Increment grayscale value
                gray_value = self.image_data[y, x]
                hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

                # Draw the pixel
                self.canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10,
                                              fill=hex_color, outline=hex_color)
            
            # Draw gray surrounding pixels
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (dx != 0 or dy != 0) and 0 <= x + dx < 28 and 0 <= y + dy < 28:
                        if self.image_data[y + dy, x + dx] == 0:  # Only if the surrounding pixel is empty
                            surrounding_gray_value = min(255, self.image_data[y + dy, x + dx] + 25)  # Lighter gray
                            self.image_data[y + dy, x + dx] = surrounding_gray_value
                            surrounding_hex_color = f'#{surrounding_gray_value:02x}{surrounding_gray_value:02x}{surrounding_gray_value:02x}'
                            
                            self.canvas.create_rectangle((x + dx) * 10, (y + dy) * 10, 
                                                          (x + dx + 1) * 10, (y + dy + 1) * 10,
                                                          fill=surrounding_hex_color, outline=surrounding_hex_color)

    def reset(self, event):
        pass

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data.fill(0)
        self.label_entry.delete(0, tk.END)

    def save_image(self):
        label = self.label_entry.get().strip()
        
        if not label.isdigit() or int(label) < 0 or int(label) > 9:
            print("Please enter a valid digit (0-9).")
            return
        
        flattened_data = self.image_data.flatten()
        df = pd.DataFrame(flattened_data).T
        df['Label'] = label

        file_mode = 'a' if os.path.exists("drawn_digit.csv") else 'w'
        df.to_csv("drawn_digit.csv", mode=file_mode, index=False, header=not os.path.exists("drawn_digit.csv"))
        
        print("Image saved to drawn_digit.csv with label:", label)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
