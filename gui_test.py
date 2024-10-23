import tkinter as tk
import numpy as np
import pandas as pd

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit")

        # Create a canvas for drawing
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack()

        # Label input for the digit
        self.label_entry = tk.Entry(master)
        self.label_entry.pack()
        self.label_entry.insert(0, "Enter digit (0-9)")

        # Create buttons
        self.save_button = tk.Button(master, text='Save', command=self.save_image)
        self.save_button.pack()
        
        self.clear_button = tk.Button(master, text='Clear', command=self.clear_canvas)
        self.clear_button.pack()

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Store drawing data (grayscale values)
        self.image_data = np.zeros((28, 28), dtype=int)

    def paint(self, event):
        x, y = event.x // 10, event.y // 10
        if 0 <= x < 28 and 0 <= y < 28:
            # Only allow drawing if the pixel is not fully black (255)
            if self.image_data[y, x] < 255:
                # Increment the grayscale value (darker)
                self.image_data[y, x] = max(self.image_data[y, x], min(100, self.image_data[y, x] + 51)) # Increase grayscale intensity
                
                # Calculate grayscale value for drawing
                gray_value = self.image_data[y, x]

                hex_color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'  # Convert to hex color

                # Draw the pixel
                self.canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10,
                                              fill=hex_color, outline=hex_color)

    def reset(self, event):
        pass

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data.fill(0)  # Reset image data
        self.label_entry.delete(0, tk.END)  # Clear the label entry

    def save_image(self):
        # Get the label from the entry
        label = self.label_entry.get().strip()
        
        # Check if the label is valid (only digits)
        if not label.isdigit() or int(label) < 0 or int(label) > 9:
            print("Please enter a valid digit (0-9).")
            return
        
        # Flatten the 2D image array and save to CSV
        flattened_data = self.image_data.flatten()
        df = pd.DataFrame(flattened_data).T  # Transpose to make it a single row
        df['Label'] = label  # Add the label to the dataframe
        df.to_csv("drawn_digit.csv", index=False, header=False)
        print("Image saved to drawn_digit.csv with label:", label)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()