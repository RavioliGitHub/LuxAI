from Games.Game import Game
import pyautogui
from selenium import webdriver
import time
from functools import partial


class DinosaurGame(Game):
    def __init__(self):
        super().__init__("DinosaurGame")
        self.commands = {
            "jump": partial(pyautogui.press, "space"),
        }

    def launch(self):
        driver = webdriver.Firefox()
        driver.get("https://offline-dino-game.firebaseapp.com/")
        time.sleep(1)
        pyautogui.click(300, 300)
        pyautogui.press("space")
