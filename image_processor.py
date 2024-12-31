 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from collections import Counter

# Determine the root directory of the current project
project_root = os.path.abspath(os.path.dirname(__file__))

def get_most_common_color(image):
    """
    Calculate the most common color in the given image.
    
    Parameters:
    image (numpy.ndarray): The input image in BGR format.
    
    Returns:
    tuple: The most common color in the image as a tuple of (B, G, R).
    """
    pixels = image.reshape(-1, 3)
    most_common_color = Counter(map(tuple, pixels)).most_common(1)[0][0]
    return most_common_color

class ImageProcessor:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.selected_base_colors = {}

    def select_image(self):
        """
        Open a file dialog to select an image.
        
        Returns:
        numpy.ndarray: The selected image in BGR format, or None if no image is selected.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path:
            return cv2.imread(file_path)
        return None

    def segment_image(self, image):
        """
        Segment the image into clusters using KMeans.
        
        Parameters:
        image (numpy.ndarray): The input image in BGR format.
        
        Returns:
        tuple: A tuple containing the segmented image and the labels of each pixel.
        """
        # Convert image from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(pixels)
        
        # Create the segmented image
        segmented_image = kmeans.cluster_centers_[labels].reshape(rgb_image.shape).astype(np.uint8)
        return segmented_image, labels.reshape(rgb_image.shape[:2])

    def find_and_draw_contours(self, image, labels):
        """
        Find and draw contours of the segmented regions.
        
        Parameters:
        image (numpy.ndarray): The input image in BGR format.
        labels (numpy.ndarray): The labels of each pixel in the image.
        
        Returns:
        numpy.ndarray: The image with contours drawn.
        """
        contours_image = image.copy()
        for label in np.unique(labels):
            mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_image, contours, -1, (0, 0, 0), 1)
        return contours_image

    def apply_color_transformation(self, image, labels, method='mean'):
        """
        Apply a color transformation to the segmented regions.
        
        Parameters:
        image (numpy.ndarray): The input image in BGR format.
        labels (numpy.ndarray): The labels of each pixel in the image.
        method (str): The method to use for color transformation ('mean', 'median', 'most_common').
        
        Returns:
        numpy.ndarray: The transformed image.
        """
        transformed_image = np.zeros_like(image)
        for label in np.unique(labels):
            mask = (labels == label)
            if method == 'mean':
                # Calculate the mean color of the region
                color = image[mask].mean(axis=0).astype(np.uint8)
            elif method == 'median':
                # Calculate the median color of the region
                color = np.median(image[mask], axis=0).astype(np.uint8)
            else:  # method == 'most_common'
                # Calculate the most common color of the region
                color = Counter(map(tuple, image[mask])).most_common(1)[0][0]
            transformed_image[mask] = color
        return transformed_image

    def closest_color(self, base_colors, color):
        """
        Find the closest base color to the given color.
        
        Parameters:
        base_colors (dict): A dictionary of base colors.
        color (tuple): The color to compare.
        
        Returns:
        int: The index of the closest base color.
        """
        base_colors_array = np.array(list(base_colors.values()))
        color = np.array(color)
        distances = np.sqrt(np.sum((base_colors_array - color) ** 2, axis=1))
        return np.argmin(distances)

    def apply_base_colors(self, image, labels, base_colors):
        """
        Apply the closest base colors to the segmented regions.
        
        Parameters:
        image (numpy.ndarray): The input image in BGR format.
        labels (numpy.ndarray): The labels of each pixel in the image.
        base_colors (dict): A dictionary of base colors.
        
        Returns:
        numpy.ndarray: The image with base colors applied.
        """
        base_color_image = np.zeros_like(image)
        for label in np.unique(labels):
            mask = (labels == label)
            mean_color = image[mask].mean(axis=0).astype(np.uint8)
            closest_base_color = list(base_colors.values())[self.closest_color(base_colors, mean_color)]
            base_color_image[mask] = closest_base_color
        return base_color_image

    def on_click(self, event, images, titles):
        """
        Display the clicked image in a new window.
        
        Parameters:
        event (matplotlib.backend_bases.MouseEvent): The click event.
        images (list): A list of images.
        titles (list): A list of titles corresponding to the images.
        """
        for i, ax in enumerate(event.canvas.figure.axes):
            if event.inaxes == ax:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(images[i])
                ax.set_title(titles[i])
                ax.axis('off')
                plt.show()
                break

    def select_base_colors(self, base_colors):
        """
        Display a dialog to select base colors.
        
        Parameters:
        base_colors (dict): A dictionary of base colors.
        """
        self.selected_base_colors = {}

        def on_submit():
            for color, var in checkboxes.items():
                if var.get() == 1:
                    self.selected_base_colors[color] = base_colors[color]
            root.destroy()

        root = tk.Tk()
        root.title("Select Base Colors")
        root.attributes('-topmost', True)
        root.focus_force()

        checkboxes = {}
        row = 0
        for color_name, color_value in base_colors.items():
            color_hex = '#%02x%02x%02x' % color_value
            var = tk.IntVar(value=1)
            checkbox = tk.Checkbutton(root, text=color_name, variable=var)
            checkbox.grid(row=row, column=0, padx=5, pady=2, sticky='w')
            preview = tk.Label(root, bg=color_hex, width=3, height=1, relief=tk.SUNKEN)
            preview.grid(row=row, column=1, padx=5, pady=2, sticky='w')
            checkboxes[color_name] = var
            row += 1

        submit_button = tk.Button(root, text="Submit", command=on_submit)
        submit_button.grid(row=row, columnspan=2, pady=10)
        root.mainloop()

    def add_colors_to_base(self, most_common_color_image, base_colors):
        """
        Add new colors from the image to the base colors.
        
        Parameters:
        most_common_color_image (numpy.ndarray): The image with the most common colors applied.
        base_colors (dict): A dictionary of base colors.
        """
        new_colors = np.unique(most_common_color_image.reshape(-1, most_common_color_image.shape[2]), axis=0)
        for color in new_colors:
            color_name = f"Color {len(base_colors) + 1}"
            base_colors[color_name] = tuple(color)


def main():
    processor = ImageProcessor()
    image = processor.select_image()

    if image is not None:
        segmented_image, labels = processor.segment_image(image)
        contours_image = processor.find_and_draw_contours(image, labels)
        mean_color_image = processor.apply_color_transformation(image, labels, 'mean')
        median_color_image = processor.apply_color_transformation(image, labels, 'median')
        most_common_color_image = processor.apply_color_transformation(image, labels, 'most_common')

        base_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "blue": (0, 0, 255),
            "brown": (165, 42, 42),
            "orange": (255, 165, 0),
            "pink": (255, 192, 203),
            "purple": (128, 0, 128),
            "tan": (210, 180, 140),
            "gray": (128, 128, 128)
        }

        processor.add_colors_to_base(most_common_color_image, base_colors)
        processor.select_base_colors(base_colors)
        base_color_image = processor.apply_base_colors(image, labels, processor.selected_base_colors)
        closest_color_image = processor.apply_color_transformation(image, labels, 'mean')
        outlined_most_common_color_image = processor.find_and_draw_contours(most_common_color_image, labels)

        images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(mean_color_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(median_color_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(most_common_color_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(outlined_most_common_color_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(base_color_image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(closest_color_image, cv2.COLOR_BGR2RGB)
        ]
        titles = [
            'Original Image', 'Outlined Image', 'Mean Color Image', 'Median Color Image',
            'Most Common Color Image', 'Outlined Most Common Color Image', 'Base Color Image', 'Closest Original Color Image'
        ]

        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        for i, axi in enumerate(ax.flat):
            if i < len(images):
                axi.imshow(images[i])
                axi.set_title(titles[i])
                axi.axis('off')

        fig.canvas.mpl_connect('button_press_event', lambda event: processor.on_click(event, images, titles))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)
        plt.show(block=True)

        root = tk.Tk()
        root.withdraw()
        if messagebox.askyesno("Save Results", "Do you want to save the results?"):
            folder_name = simpledialog.askstring("Input", "Enter a name for the folder:")
            if folder_name:
                folder_path = os.path.join(project_root, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                for img, title in zip(images, titles):
                    image_path = os.path.join(folder_path, f"{title.replace(' ', '_')}.png")
                    cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                print("Save operation cancelled.")
        else:
            print("Program ended without saving.")
    else:
        print("No image selected.")

if __name__ == "__main__":
    main() 
