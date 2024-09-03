import pyautogui
import datetime

class Game():
    def __init__(self, name):
        self.name = name

    def launch(self):
        raise NotImplementedError

    def commands(self):
        raise NotImplementedError

    def screenshot(self, region):
        """
        Do screenshot and save with datetime
        :return:
        """
        screenshot = pyautogui.screenshot(region=region)
        file_name = f"Screenshots/{self.name}{datetime.datetime.now()}.png"
        screenshot.save(file_name)
        # pyautogui.screenshot(f"Screenshots/{self.name}{datetime.datetime.now()}.png")

        return file_name
