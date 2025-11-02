import cv2
import matplotlib.pyplot as plt

def main():
    images = []
    image = load_image("shareImage/cat.jpg", title="Cat")
    images.append(image[0])
    display_image(image=image[0], title=image[1].__str__())
    pass



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

if __name__ == "__main__":
    main()
    print("Done")