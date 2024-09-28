#! python3
# mouseNow.py - Displays the mouse cursor's current position.
import pyautogui
from pynput import mouse
import time

clic_data = []


def on_click(x, y, button, pressed):
    if not pressed and button == mouse.Button.middle:
        print("")

def print_mouse_position():
    """
    Continuously displays the mouse coordinates
    and the color of the current pixel
    """
    try:
        while True:
            x, y = pyautogui.position()
            pixel_color = pyautogui.screenshot().getpixel((x, y))

            """
            position_str = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            position_str += ' RGB: (' + str(pixel_color[0]).rjust(3)
            position_str += ', ' + str(pixel_color[1]).rjust(3)
            position_str += ', ' + str(pixel_color[2]).rjust(3) + ')'
            """
            position_str = "pyautogui.pixelMatchesColor(x="
            position_str += str(x)
            position_str += ", y="
            position_str += str(y)
            position_str += ", expectedRGBColor="
            position_str += str(pixel_color)
            position_str += ")"

            print('\r', end='', flush=True)  # deletes the previous line
            print(position_str, end='', flush=True)

    except KeyboardInterrupt:
        print('\nDone.')


def main():
    mouse_thread = mouse.Listener(
        on_click=on_click,
        on_scroll=None,
        on_move=None
    )

    mouse_thread.start()
    print_mouse_position()



main()