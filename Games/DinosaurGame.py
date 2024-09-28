from Games.Game import Game
import pyautogui
from selenium import webdriver
import time
from functools import partial
import numpy as np
from PIL import ImageGrab, ImageOps


class DinosaurGame(Game):
    def __init__(self):
        super().__init__("DinosaurGame")
        self.commands = {
            "jump": partial(pyautogui.press, "space"),
        }

    def launch(self):
        driver = webdriver.Firefox()
        driver.set_window_rect(500, 100, 200, 400)
        driver.get("https://offline-dino-game.firebaseapp.com/")
        time.sleep(1)
        pyautogui.click(600, 300)
        pyautogui.press("space")

    def screenshot_region(self):
        return 50, 50, 52, 52

    def screenshot_box(self):
        return (self.screenshot_region()[0],
                self.screenshot_region()[1],
                self.screenshot_region()[0] + self.screenshot_region()[2],
                self.screenshot_region()[1] + self.screenshot_region()[3])


    def analyze(self):
        data = {
            #"is_dead": self.is_dead()
            "is_dead": False
        }
        # grabbing all the pixels values in form of RGB tuples
        image = ImageGrab.grab(self.screenshot_box())

        # converting RGB to Grayscale to
        # make processing easy and result faster
        grayImage = ImageOps.grayscale(image)

        # using numpy to get sum of all grayscale pixels
        a = np.array(grayImage.getcolors())
        print(a)

        return data

    def is_dead(self):
        im = self.screenshot()
        points = [(766, 581), (764, 540), (925, 472), (975, 471)]
        return all([im.getpixel((100, 200)) == (83, 83, 83, 255) for (x, y) in points])
