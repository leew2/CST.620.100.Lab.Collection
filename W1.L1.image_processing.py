import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = load_image("shareImage/cat.jpg")
    display_image(image, "Cat")


def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image.")
        return None
    return image

def display_image(image, title="Image"):
    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()


        
    else:
        print("No image to display.")


if __name__ == "__main__":
    main()
    print("Image processing module executed successfully.")