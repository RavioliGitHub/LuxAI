from Games.DinosaurGame import DinosaurGame
import time
from TextRecognition.Tesseract import Tesseract

text_reader = Tesseract()
game = DinosaurGame()
game.launch()
time.sleep(2)

print("Starting loop")
while True:
    start = time.time()
    #game.screenshot()
    print(game.analyze())
    print('\r', end='', flush=True)  # deletes the previous line
    print(f"Time elapsed: {time.time() - start}", end='', flush=True)
