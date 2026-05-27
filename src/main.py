
import sys

from includes.eval_utils import eval
from includes.train_utils import train

if __name__ == "__main__":
    """Executable with both the capability of tranining models and use those models to find shapes on input images"""

    if len(sys.argv) == 1 or '-h' in sys.argv or '"-help"' in sys.argv:
        print("Use '-train' to set the mode to train a classification model.")
        print("Use '-eval' to set the mode to classify a group of images based on a previously trained model.")
        print("Use '-src' when training followed by a folder containing images to be used for training the model, or, when evaluating, followed by an image or a folder containing images to be classified.")
        print("Use '-model' when evaluating followed by the path to a previously trained model file.")
        print("Use '-dst' followed by the path to the destination model '.pth' file when training, or, when evaluating, followed by the path to the classification '.csv' file.")
        print("Use '-usr' when evaluating to open a drawing area, where the user may draw an input image. This option will prevail over '-src'. On the drawing area use: LMB to draw, RMB to erase, 'c' to clear the canvas and 'a' to accept the image and proceed.")
        print("Use '-h' for help.")
        print("Add '-ow' to overwrite the file specified after '-dst', if it already exists.")
        print("Add '-d' to diagnose what the algorithms are doing under the hood.")
        print("Add '-cls' when evaluating, followed by a threshold between 0-1 to write the best classification instead of match scores to the output file (default value is 0.1).")
        sys.exit(0)

    if '-train' in sys.argv:
        train(sys.argv)
    elif '-eval' in sys.argv:
        eval(sys.argv)
    else:
        print(f"No single mode defined on the argument list ('-train' or '-eval')") 
        sys.exit(-1)

    print("Finished")
    sys.exit(0)