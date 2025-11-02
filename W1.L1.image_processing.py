import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img_compare = []
    image = load_image("shareImage/cat.jpg", title="Cat")
    img_compare.append(image[0])
    display_image(image=image[0], title=image[1].__str__())
    img_compare.clear()
    for kernel_size in [(5, 5),(11, 11),(15, 15)]:
        guas = display_gaussian_blur(image=image[0], title='Gaussian Blur', kernel_size=kernel_size, display=False)
        img_compare.append(guas[0])
    display_image(img_compare, title="gaussian blur with different kernel sizes")
    img_compare.clear()
    for method in ["Canny", "Sobel"]:
        edges = edge_detection(image=image[0], title="Cat", method=method, display=False)
        img_compare.append(edges[0])
    display_image(img_compare, title="Edge Detection")
    img_compare.clear()

    
def load_image(file_path, title="Image"):
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image.")
        return None, None
    return image, title

def display_image(image, title="Image"):
    if image is not None:
        if not isinstance(image, list):
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
            plt.show()
        elif isinstance(image, list):
            for idx, img in enumerate(image):
                plt.subplot(1, len(image), idx + 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"{title} {idx+1}")
                plt.axis("off")
            plt.show()
        else:
            print("Invalid image format for display.")
    else:
        print("No image to display.")

def display_gaussian_blur(image, title="Image", kernel_size=(5, 5), sigma=0, display=True):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    if display:
        display_image(blurred_image, f"{title} - Gaussian Blur")
    return blurred_image, f"{title} - Gaussian Blur", kernel_size, sigma

def edge_detection(image, title="Image", method="Canny", display=True):
    if method == "Canny":
        edges = cv2.Canny(image, 100, 200)
    elif method == "Sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)
    else:
        print("Unsupported edge detection method.")
        return None, None
    if display:
        display_image(edges, f"{title} - {method} Edge Detection")
    return edges, f"{title} - {method} Edge Detection", method


if __name__ == "__main__":
    main()
    print("Image processing module executed successfully.")