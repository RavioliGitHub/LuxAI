import sys
from PyQt5 import QtWidgets, QtCore
import pyautogui
from PIL import Image
import numpy as np
from PIL import ImageGrab, ImageOps


class FrameLine(QtWidgets.QMainWindow):
    def __init__(self, x, y, width, height, parent_frame, update_callback=None, orientation=None):
        super(FrameLine, self).__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.parent_frame = parent_frame
        self.setGeometry(x, y, width, height)
        self.update_callback = update_callback
        self.orientation = orientation  # 'horizontal' for top/bottom, 'vertical' for left/right
        self.mouse_is_pressed = False
        self.last_position = None
        self.setStyleSheet("border: 20px solid green;")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_is_pressed = True
            self.last_position = event.globalPos()
        if event.button() == QtCore.Qt.RightButton:
            self.parent_frame.take_screenshot()

    def mouseMoveEvent(self, event):
        if self.mouse_is_pressed:
            delta = event.globalPos() - self.last_position

            if self.orientation == 'horizontal':
                # Move only up or down
                self.move(self.x(), self.y() + delta.y())
            elif self.orientation == 'vertical':
                # Move only left or right
                self.move(self.x() + delta.x(), self.y())

            self.last_position = event.globalPos()

            # Update other frames if necessary
            if self.update_callback:
                self.update_callback(self)

    def mouseReleaseEvent(self, event):
        self.mouse_is_pressed = False


class Frame:
    def __init__(self, x, y, width, height):
        app = QtWidgets.QApplication(sys.argv)

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.frame_thickness = 10

        # Create frames around the specified area
        self.top_frame = FrameLine(x, y - self.frame_thickness, width, self.frame_thickness, self, self.update_frames, 'horizontal')
        self.bottom_frame = FrameLine(x, y + height, width, self.frame_thickness, self, self.update_frames, 'horizontal')
        self.left_frame = FrameLine(x - self.frame_thickness, y, self.frame_thickness, height, self, self.update_frames, 'vertical')
        self.right_frame = FrameLine(x + width, y, self.frame_thickness, height, self, self.update_frames, 'vertical')

        # Show all frames
        self.top_frame.show()
        self.bottom_frame.show()
        self.left_frame.show()
        self.right_frame.show()

        sys.exit(app.exec_())

    def update_frames(self, changed_frame):
        x = self.left_frame.x() + self.frame_thickness
        y = self.top_frame.y() + self.frame_thickness
        width = self.right_frame.x() - self.left_frame.x() - self.frame_thickness
        height = self.bottom_frame.y() - self.top_frame.y() - self.frame_thickness

        if changed_frame == self.left_frame or changed_frame == self.right_frame:
            self.top_frame.setGeometry(x, self.top_frame.y(), width, self.frame_thickness)
            self.bottom_frame.setGeometry(x, self.bottom_frame.y(), width, self.frame_thickness)

        if changed_frame == self.top_frame or changed_frame == self.bottom_frame:
            self.left_frame.setGeometry(self.left_frame.x(), y, self.frame_thickness, height)
            self.right_frame.setGeometry(self.right_frame.x(), y, self.frame_thickness, height)

        self.x = self.top_frame.geometry().x()
        self.y = self.right_frame.geometry().y()
        self.width = self.top_frame.geometry().width()
        self.height = self.right_frame.geometry().height()

    def take_screenshot(self):
        pyautogui.screenshot("test.png", region=(self.x, self.y, self.width, self.height))
        print(self.x, self.y, self.width, self.height)
        img = Image.open("test.png")
        # Display the image
        #img.show()
        # grabbing all the pixels values in form of RGB tuples
        image = ImageGrab.grab((self.x, self.y, self.x + self.width, self.y + self.height))

        # converting RGB to Grayscale to
        # make processing easy and result faster
        grayImage = ImageOps.grayscale(image)

        # using numpy to get sum of all grayscale pixels
        a = np.array(grayImage.getcolors())
        print(a)

# Given initial coordinates and size for the area to frame
x, y, width, height = 260, 260, 200, 200
Frame(x, y, width, height)

