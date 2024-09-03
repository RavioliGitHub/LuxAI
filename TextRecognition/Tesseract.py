from PIL import Image
import pytesseract


class Tesseract:
    def __init__(self):
        pass

    @staticmethod
    def get_text(image):
        image = Image.open(image)
        # Use pytesseract to do OCR on the image
        return pytesseract.image_to_string(image)
