import cv2
from matplotlib import pyplot as plt
import re
import logging
from paddleocr import PaddleOCR
from parameters import parameters_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

image_path = r'your_input_path'
vconf = 0.8

def apply_container_rules(text):
    if len(text) == 11 and len(re.findall(r'\d', text)) > 7:
        first_four = text[:4].replace('0', 'O')
        corrected_text = first_four + text[4:]
        return corrected_text
    return text

def replace_special_symbols(text):
    text = re.sub(r'\s+', '', text) 
    return text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

class OCRProcessor:
    def __init__(self):
        """Initialize PaddleOCR"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    def show_image(self, title, image, title2, image2, cmap='gray'):
        """Display images before and after processing"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image, cmap=cmap)
        ax[0].set_title(title)
        ax[0].axis('off')

        ax[1].imshow(image2, cmap=cmap)
        ax[1].set_title(title2)
        ax[1].axis('off')

        plt.show()

    def show_ocr_results(self):
        """Display OCR results"""
        if not self.result:
            print("No OCR results")
            return None

        combine_text = ""
        
        for _,(text,score) in self.result:
            text = replace_special_symbols(text)
            text = clean_text(text)
            score = round(score, 2)
            if score >= vconf:
                if len(combine_text + text) == 11:
                    combine_text += text
                else:
                    print("Invalid container ID")
                    break
            else:
                print(f'Low Confidence {round(score,2)}')
                
        combine_text = apply_container_rules(combine_text)

        if re.fullmatch(r'[A-Z]{4}\d{7}$', combine_text):
            print(f"Result : {combine_text}")
        else:
            print("Unable to detect ID")
    
    def ocr_read(self, original_image, image_processing):
        """Read text from image"""
        self.result = self.ocr.ocr(image_processing, cls=True)[0]
        self.show_ocr_results()
        self.show_image('original', original_image, 'processing', image_processing)

try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if image is not None:
        processor = OCRProcessor()
        parameters = parameters_processor()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret = parameters.rotate_image(gray)
        rm_n = parameters.remove_noise(ret)
        es = parameters.enhance_sharpness(rm_n)
        eb = parameters.enhance_brightness(es)
        processor.ocr_read(image, eb)
    else:
        print("Image not found")
except Exception as e:
    logging.error(f"Error during OCR processing: {e}")
