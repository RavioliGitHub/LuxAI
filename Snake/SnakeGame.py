import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
from Snake.model import SnakeNet


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x, y")

# we substitute pygame.init() for the explicit init of inititng the display and fonts
# this will prevent sound init, which may cause issues on notebooks
pygame.display.init()
pygame.font.init()

font = pygame.font.Font(pygame.font.get_default_font(), 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 120

class SillySnakeGameAi:

    def __init__(self, width=320, height=240, playerName="Player"):
        self.w = width
        self.h = height
        self.playerName = playerName

        # init display

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Silly Snake")
        self.clock = pygame.time.Clock()

        # init game
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
                      Point(self.head.x - (3 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None

        self.placeFood()

        self.frameIteration = 0


    def placeFood(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.placeFood()

    def playStep(self, action):
        # increment frame iteration

        self.frameIteration += 1

        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move the snake, asta creste si marimea sarpelui
        self.moveSnake(action)

        # reward stuff
        reward = 0

        # 3. check if game over
        # TODO: Ici il me semble qu'il faut retirer le fait d'avoir une pénalité quand on des frames qui atteignent un seuil
        if self.isCollision():
            gameOver = True

            reward -= 1

            return reward, gameOver, self.score

        # 4. move snake or place food

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.placeFood()
        else:
            self.snake.pop()

        # Check if game over based on max number of iterations
        if self.frameIteration > 100 * len(self.snake):
            gameOver = True

            return reward, gameOver, self.score

        # 5. update ui and clock
        self.updateUi()
        self.clock.tick(SPEED)
        # 6. return game over and score
        gameOver = False

        return reward, gameOver, self.score

    def isCollision(self, p: Point = None):

        if p == None:
            p = self.head

        #check if it hits border
        if p.x > self.w - BLOCK_SIZE or p.x < 0:
            return True

        if p.y > self.h - BLOCK_SIZE or p.y < 0:
            return True

        # check if it hits itself

        if p in self.snake[1:]:
            return True

        return False

    def moveSnake(self, action):

        # action -> [straigth, right, left]
        clockWiseDirections = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        currentDirectionIndex = clockWiseDirections.index(self.direction)

        newDirection = self.direction

        if np.array_equal(action, [0, 1, 0]): # Turn right
            newDirection = clockWiseDirections[(currentDirectionIndex + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]): # Turn left
            newDirection = clockWiseDirections[(currentDirectionIndex - 1) % 4]

        self.direction = newDirection

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        # this grows the size of our snake
        self.snake.insert(0, self.head)


    def updateUi(self):
        self.display.fill(BLACK)

        for p in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(p.x + 4, p.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        scoreText = font.render("Score: " + str(self.score) + " Speed: " + str(SPEED) + " Player: " + self.playerName, True, WHITE)

        self.display.blit(scoreText, [0, 0])

        pygame.display.flip()


    def setPlayerName(self, name):
        self.playerName = name

    def get_game_grid(self):
        """0: Empty space
            1: Food
            2: Snake body
            3: Snake head"""
        # Create grid
        grid = torch.zeros((self.h//BLOCK_SIZE, self.w//BLOCK_SIZE))
        # Add food
        grid[int(self.food.y//BLOCK_SIZE), int(self.food.x//BLOCK_SIZE)] = 1
        # Add snake body
        for segment in self.snake:
            grid[int(segment.y // BLOCK_SIZE), int(segment.x // BLOCK_SIZE)] = 2
        # Add snake head
        grid[int(self.head.y // BLOCK_SIZE), int(self.head.x // BLOCK_SIZE)] = 3

        return grid

def playGame(playerName = "Stefan"):
    # Create game
    game = SillySnakeGameAi(playerName=playerName)
    # Create model
    model = SnakeNet(int(game.w//BLOCK_SIZE), int(game.h//BLOCK_SIZE), 3)

    while True:
        # Get game state
        grid = game.get_game_grid()
        print(grid)
        grid = grid.unsqueeze(0).unsqueeze(0)
        # Get next action
        # straigth [1, 0, 0]
        # right [0, 1, 0]
        # left [0, 0, 1]
        """random_number = np.random.randint(0, 3)"""
        """
        Batch dimension: If you have only one grid, you can add a batch dimension with size 1.
Channels dimension: Since you are working with a single channel (a 2D grid), you need to add a channel dimension, which will also have size 1.
        """
        # Get action from the nn
        decision = model.forward(grid)
        # Highest Q value correspond to the action
        # Get the index of the highest value
        max_index = torch.argmax(decision).item()
        action = [0, 0, 0]
        action[max_index] = 1
        reward, gameOver, score = game.playStep(action)


        # break if game over

        if gameOver == True:
            break

    print("Game Over, final score: ", score)

    pygame.quit()

# include a main function, to be able to run the script, for notebooks we can comment this out
if __name__ == "__main__":
    playGame()
