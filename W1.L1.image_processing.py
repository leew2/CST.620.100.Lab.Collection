import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img_compare = []
    image = load_image("shareImage/cat.jpg", title="Cat")
    img_compare.append(("Original Image", image))
    #display_image(image, "Cat")

    for kernel_size in [(3, 3), (5, 5), (7, 7)]:
        guas = display_gaussian_blur(image, "Cat", kernel_size=kernel_size, display=False)
        img_compare.append((f"Gaussian Blur {kernel_size}", guas))

    

def load_image(file_path, title="Image"):
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image.")
        return None, None
    return image, title

def display_image(image, title="Image"):
    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        print("No image to display.")

def display_gaussian_blur(image, title="Image", kernel_size=(5, 5), sigma=1.5, display=True):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    if display:
        display_image(blurred_image, f"{title} - Gaussian Blur")
    return blurred_image, f"{title} - Gaussian Blur", kernel_size, sigma

if __name__ == "__main__":
    main()
    print("Image processing module executed successfully.")