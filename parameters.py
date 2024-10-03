import cv2
import numpy as np
from PIL import Image, ImageEnhance

class parameters_processor:
    def rotate_image(self, image):
        height, width = image.shape[:2]

        if height > width:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return rotated_image
        else:
            return image

    def remove_noise(self, image):
        """Remove noise by blurring"""
        noise_removed = cv2.GaussianBlur(image, (1, 1), 0)
        return noise_removed
    
    def enhance_sharpness(self, image):
        """Enhance sharpness of the image"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced_image = enhancer.enhance(2)
        return np.array(enhanced_image)

    def enhance_brightness(self, image):
        """Enhance brightness of the image"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_image = enhancer.enhance(1.1)
        return np.array(enhanced_image)