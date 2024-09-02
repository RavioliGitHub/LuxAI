import pyautogui
import datetime

class Game():
    def __init__(self, name):
        self.name = name

    def launch(self):
        raise NotImplementedError

    def commands(self):
        raise NotImplementedError

    def screenshot(self):
        """
        Do screenshit and save with datetime
        :return:
        """
        pyautogui.screenshot(f"Screenshots/{self.name}{datetime.datetime.now()}.png")

