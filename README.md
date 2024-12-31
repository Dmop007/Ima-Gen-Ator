# Ima-Gen-Ator
Image processor that looks for areas with similar colors and makes them all the same.

Image Processing with Color Segmentation and Transformation

This project provides a script for image processing using OpenCV, focusing on color segmentation and transformation. The script allows users to select an image, process it by examining areas of similar color, and change colors to a single representative color for each area. The transformations include applying mean, median, and most common color methods.
Features

    Color Segmentation: Segments the image into clusters using KMeans clustering.
    Color Transformations: Applies mean, median, and most common color transformations to the image.
    Base Colors Application: Applies the closest base colors to the segmented regions.
    Contours Drawing: Finds and draws contours around the segmented regions.
    Interactive Visualization: Displays the processed images with an option to save the results.

Requirements

    Python 3.x
    OpenCV
    NumPy
    Matplotlib
    Scikit-learn
    Tkinter (usually included with Python)

Installation

    Clone the Repository:
    sh

git clone https://github.com/your-username/your-repository.git
cd your-repository

Install Dependencies:

You can install the required dependencies using pip:

pip install opencv-python-headless numpy matplotlib scikit-learn

Usage

    Run the Script:

    python image_processor.py

    Select an Image:
        A file dialog will appear. Select the image you want to process.

    Process the Image:
        The script will process the image and display various transformations, including mean color, median color, and most common color segmentation.

    Interact with the Images:
        Click on any displayed image to view it in a larger window.

    Save the Results:
        A prompt will appear asking if you want to save the results. Enter a folder name to save the processed images.

Files

    image_processor.py: Main script for image processing.

Example

Here's an example of how to use the script:

    Run the Script:

    python image_processor.py

    Select an Image:

    View Processed Images:

License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
Contact

For any questions or inquiries, please contact the project maintainer at [dmop007@gmail.com].
