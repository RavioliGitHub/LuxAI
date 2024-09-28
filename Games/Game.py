import pyautogui
import datetime

class Game():
    def __init__(self, name):
        self.name = name

    def launch(self):
        raise NotImplementedError

    def commands(self):
        raise NotImplementedError

    def screenshot_region(self):
        raise NotImplementedError

    def screenshot(self, save=True):
        file_name = f"Screenshots/{self.name}{datetime.datetime.now()}.png"
        if save:
            screenshot = pyautogui.screenshot(file_name, region=self.screenshot_region())
        else:
            screenshot = pyautogui.screenshot(region=self.screenshot_region())
        return screenshot

    def analyze(self):
        raise NotImplementedError
