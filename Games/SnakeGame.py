import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch


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
GREEN = (0, 255, 0)
GREEN2 = (0, 128, 0)
CRIMSON = (220, 20, 60)

BLOCK_SIZE = 20
SPEED = 120


# TODO: Changer la valeur des rewards et penlaties, elles devraient être entre 0 et 1.
class SillySnakeGameAi:
    def __init__(self, width=16, height=12, playerName="Player"):
        self.w = width*BLOCK_SIZE
        self.h = height*BLOCK_SIZE
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
        self.distance_to_food = None
        self.food = None

        self.placeFood()

        self.frameIteration = 0


    def placeFood(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.placeFood()
        # Compute distance between snake and food
        self.distance_to_food = (abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)) // BLOCK_SIZE

    def play_step(self, action):
        # Prepare the one hot encoding of the action
        action_onehot = [0, 0, 0]
        action_onehot[action] = 1

        # Increment frame iteration
        self.frameIteration += 1

        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                quit()

        # Reward variable associated to the play step
        reward = 0

        old_distance_to_food = self.distance_to_food
        # 2. move the snake, asta creste si marimea sarpelui
        self.moveSnake(action_onehot)
        # Update distance from snake to food
        self.distance_to_food = (abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)) // BLOCK_SIZE


        # 3. check if game over
        if self.isCollision():
            gameOver = True

            reward -= 1

            return reward, gameOver

        # 4. move snake or place food

        if self.head == self.food:
            self.score += 1
            reward = 1
            self.placeFood()
        else:
            self.snake.pop()
            # Reward when getting closer to food
            if old_distance_to_food >= self.distance_to_food:
                reward += 0.1
            else:
                # Penalty for useless steps
                reward -= 0.2

        # Check if game over based on max number of iterations
        if self.frameIteration > 100 * len(self.snake):
            gameOver = True

            return reward, gameOver

        # 5. update ui and clock
        self.updateUi()
        self.clock.tick(SPEED)
        # 6. return game over and score
        gameOver = False

        return reward, gameOver

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

        # Update direction
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

        # Draw Snake body
        for p in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(p.x + 4, p.y + 4, 12, 12))

        # Draw Snake eyes and tongue
        if self.direction == Direction.LEFT:
            # Eyes
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + 1, self.head.y + 1, 6, 6))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + 1, self.head.y + BLOCK_SIZE - 7, 6, 6))
            # Tongue
            pygame.draw.rect(self.display, CRIMSON, pygame.Rect(self.head.x - 6, self.head.y + BLOCK_SIZE//2 - 3, 6, 6))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x - 9, self.head.y + BLOCK_SIZE // 2 - 6, 3, 3))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x - 9, self.head.y + BLOCK_SIZE // 2 + 3, 3, 3))
        elif self.direction == Direction.RIGHT:
            # Eyes
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + BLOCK_SIZE - 7, self.head.y + 1, 6, 6))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + BLOCK_SIZE - 7, self.head.y + BLOCK_SIZE - 7, 6, 6))
            # Tongue
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE, self.head.y + BLOCK_SIZE // 2 - 3, 6, 6))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE + 6, self.head.y + BLOCK_SIZE // 2 - 6, 3, 3))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE + 6, self.head.y + BLOCK_SIZE // 2 + 3, 3, 3))
        elif self.direction == Direction.UP:
            # Eyes
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + 1, self.head.y + 1, 6, 6))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + BLOCK_SIZE - 7, self.head.y + 1, 6, 6))
            # Tongue
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 - 3, self.head.y - 6, 6, 6))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 - 6, self.head.y - 9, 3, 3))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 + 3, self.head.y - 9, 3, 3))
        else: # Down
            # Eyes
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + 1, self.head.y + BLOCK_SIZE - 7, 6, 6))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.head.x + BLOCK_SIZE - 7, self.head.y + BLOCK_SIZE - 7, 6, 6))
            # Tongue
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 - 3, self.head.y + BLOCK_SIZE + 1, 6, 6))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 - 6, self.head.y + BLOCK_SIZE + 6, 3, 3))
            pygame.draw.rect(self.display, CRIMSON,
                             pygame.Rect(self.head.x + BLOCK_SIZE // 2 + 3, self.head.y + BLOCK_SIZE + 6, 3, 3))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        scoreText = font.render("Score: " + str(self.score) + " Speed: " + str(SPEED) + " Player: " + self.playerName, True, WHITE)

        self.display.blit(scoreText, [0, 0])

        pygame.display.flip()

    def setPlayerName(self, name):
        self.playerName = name

    # TODO: should add the direction ! Or something similar, otherwise the snake cannot learn what is left, straight
    #  or right. --> Problem, how to add the direction ? Cannot be a number in front of the snake, otherwise, how will
    #  the overlap between the elements be considered --> should probably just allow all directions, unrelative to the
    #  snake --> maybe a different number of the neck (snake body block right after the snake head ?)
    def get_game_grid(self):
        """0: Empty space
            1: Food
            2: Snake body
            3: Snake neck
            4: Snake head"""
        # Create grid
        grid = torch.zeros((self.h//BLOCK_SIZE, self.w//BLOCK_SIZE))
        # Add food
        grid[int(self.food.y//BLOCK_SIZE), int(self.food.x//BLOCK_SIZE)] = 1
        # Add snake body
        for segment in self.snake[2:]:
            grid[int(segment.y // BLOCK_SIZE), int(segment.x // BLOCK_SIZE)] = 2
        # Add snake neck
        grid[int(self.snake[1].y // BLOCK_SIZE), int(self.snake[1].x // BLOCK_SIZE)] = 3
        # Add snake head
        grid[int(self.head.y // BLOCK_SIZE), int(self.head.x // BLOCK_SIZE)] = 4

        # Normalize the grid values to [0, 1] by dividing by the max value (4)
        grid = grid / 4.0

        return grid

    def get_simplified_game_state(self):
        # Get points coordinates around snake head
        point_left = Point(self.head.x - BLOCK_SIZE, self.head.y)
        point_right = Point(self.head.x + BLOCK_SIZE, self.head.y)
        point_up = Point(self.head.x, self.head.y - BLOCK_SIZE)
        point_down = Point(self.head.x, self.head.y + BLOCK_SIZE)

        # Get snake's current direction
        direction_left = self.direction == Direction.LEFT
        direction_right = self.direction == Direction.RIGHT
        direction_up = self.direction == Direction.UP
        direction_down = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and self.isCollision(point_right)) or
            (direction_left and self.isCollision(point_left)) or
            (direction_up and self.isCollision(point_up)) or
            (direction_down and self.isCollision(point_down)),

            # Danger right
            (direction_up and self.isCollision(point_right)) or
            (direction_down and self.isCollision(point_left)) or
            (direction_left and self.isCollision(point_up)) or
            (direction_right and self.isCollision(point_down)),

            # Danger left
            (direction_down and self.isCollision(point_right)) or
            (direction_up and self.isCollision(point_left)) or
            (direction_right and self.isCollision(point_up)) or
            (direction_left and self.isCollision(point_down)),

            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        state = [1 if x else 0 for x in state]

        state = torch.tensor(state).view(1, len(state)).float()

        return state
