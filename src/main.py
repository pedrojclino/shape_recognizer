import cv2
import numpy as np
import sys
import os


from includes.solve_utils import solve
from includes.train_utils import train


# INFO: Compilable in /src with:
# > py -m PyInstaller --onefile --icon=../res/icons/app_icon.ico --distpath ../bin --workpath ../build --specpath ../build --name shape_recognizer main.py

if __name__ == "__main__":
    """Executable with both the capability of traning shapes and finding them from given images"""
    
    if len(sys.argv) == 1 or '-h' in sys.argv or '"-help"' in sys.argv:
        print("Use '-train' to run the executable for training.")
        print("Use '-solve' to run th executable for finding shapes.")
        print("Use '-src' followed by a path to an image or a folder containing images to be used for training or for solving.")
        print("Use '-model' when solving followed by the path to a previously trained model file.")
        print("Use '-dst' followed by the path to the destination destination .csv file.")
        print("Use '-usr' when solving to draw an image to be searched for trained shapes (while drawing, use 'c' to clear and 'a' to accept).")
        print("Use '-h' for help.")
        print("Add '-ow' to overwrite the previous file specified as -dst if it already exists.")
        print("Add '-v' when solving to show the shape finding results.")
        print("Add '-d' to debug the shapes being trained.")
        print("Add '-s' to specify the shape search sensitivity, it's default value is 0.01 and is truncated between 0 and 0.1.")
        sys.exit(0)

    # Set mode 
    if ('-train' not in sys.argv and '-solve' not in sys.argv) or \
       ('-train' in sys.argv and '-solve' in sys.argv):
        print(f"No single mode defined on the argument list ('-train' or '-solve')") 
        sys.exit(-1)

    if '-train' in sys.argv:
        train(sys.argv)
    elif '-solve' in sys.argv:
        solve(sys.argv)

    print("Finished")
    sys.exit(0)