import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    images = []
    image = load_image("shareImage/cat.jpg")
    images.append(image)
    display_image(image=images[0], title="Cat")
    rotate = rotate_image(image=images[0], angle=45)
    images.append(rotate)
    display_image(image=images[1], title="Rotated Image")
    affine = affine_transform(image=images[0])
    images.append(affine)
    display_image(image=images[2], title="Affine Transformed Image")
    pass

def affine_transform(image, M=None, output_size=None):
    if M is None:
        start_points = np.array([[50, 50], [200, 50], [50, 200]]).astype(np.float32)
        end_points = np.array([[10, 100], [200, 50], [100, 250]]).astype(np.float32)
        M = cv2.getAffineTransform(start_points, end_points)
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])
    transformed = cv2.warpAffine(image, M, output_size)
    return transformed

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image.")
        return None
    return image

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

if __name__ == "__main__":
    main()
    print("Done")