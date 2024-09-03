from Games.DinosaurGame import DinosaurGame
import time
from TextRecognition.Tesseract import Tesseract

text_reader = Tesseract()
game = DinosaurGame()
game.launch()
time.sleep(2)

while True:
    screenshot_path = game.screenshot(region=(50, 320, 800, 300))
    if "GAME" in text_reader.get_text(screenshot_path):
        break
