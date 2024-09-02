from Games.DinosaurGame import DinosaurGame
import time

game = DinosaurGame()
game.launch()


for i in range(5):
    time.sleep(1)
    game.commands["jump"]()
    game.screenshot()
