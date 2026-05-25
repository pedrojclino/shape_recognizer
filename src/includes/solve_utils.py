import sys
import os
import cv2

from .descriptor_utils import ImageQualification

from .file_utils import save_result_file, read_model_file

from .drawing_utils import user_drawing

from .plot_utils import plot_confusion_matrix


def solve(argv):
    """Creates a csv file with the confusion matrix of the qualification of shapes on the input images based on the model file"""

    # Validate destination
    if '-dst' not in argv:
        print(f"No '-dst' found on the argument list")
        sys.exit(-1)

    idx = argv.index('-dst')
    if idx+1 >= len(argv):
        print(f"No path set after '-dst'") 
        sys.exit(-1)

    dst_file_path = argv[idx+1]
    
    texts = os.path.splitext(dst_file_path)
    if len(texts) <= 1 or texts[1] != ".csv":
        print(f"Path specified after '-dst' is not a .csv.")
        sys.exit(-1)

    # Validate model
    if '-model' not in argv:
        print(f"No training model file set to be used for solving.") 
        sys.exit(-1)

    idx = argv.index('-model')
    if idx+1 >= len(argv):
        print(f"No path set after '-model'.") 
        sys.exit(-1)

    texts = os.path.splitext(argv[idx+1])
    if len(texts) <= 1 or texts[1] != ".json":
        print(f"Path specified after '-model' is not a .json.")
        sys.exit(-1)

    # Read the model
    model_file_path = argv[idx+1]
    known_shape_descriptors = read_model_file(model_file_path)

    if not known_shape_descriptors:
        print(f"Failed to read valid model data! Please check '{model_file_path}' file integrity.")
        sys.exit(-1)

    # Validate the input
    if ('-usr' not in argv and '-src' not in argv) or \
       ('-usr' in argv and '-src' in argv):
        print(f"No single input found on the argument list ('-src' or '-usr' ).")
        sys.exit(-1)

    input_imgs = dict()
    if '-usr' in argv:
        input_imgs["user"] = user_drawing()

    elif '-src' in argv:

        idx = argv.index('-src')
        if idx +1 >= len(argv) or not os.path.exists(argv[idx+1]):
            print(f"Failed to find the specified path after '-src'.")
            sys.exit(-1)
        
        input_path = argv[idx+1]
    
        # For a single image input
        if os.path.isfile(input_path):
            img_file_name = os.path.splitext(os.path.basename(input_path))[0]
            try:
                input_imgs[img_file_name] = cv2.imread(input_path)
            except Exception as e:
                pass

        # For folder with multiple images input
        elif os.path.isdir(input_path):

            img_file_basenames = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))] 
            for img_file_basename in img_file_basenames:
                full_img_path = os.path.join(input_path, img_file_basename)
                img_file_name = os.path.splitext(img_file_basename)[0]
                try:
                    input_imgs[img_file_name] = cv2.imread(full_img_path)
                except Exception as e:
                    pass

    
    if not input_imgs:
        print(f"Unable to accept the inputs for training. Use '-h' for details on how to run the executable.")
        sys.exit(-1)

    # Set debug active
    debug = True if '-d' in argv else False

    # Set sensitivity value
    sensitivity = 0.01
    if '-s' in argv:
        idx = argv.index('-s')
        if idx+1 >= len(argv):
            print(f"No sensitivity set after '-s'.") 
            sys.exit(-1)

        if 0 < float(argv[idx+1]) < 0.1:
            sensitivity = float(argv[idx+1])
        else:
            print(f"Invalid sensitivity value. Fallback to 0.01.") 
            sensitivity = 0.01

    # Qualify images
    image_qualifications = list()
    for name, img in input_imgs.items():
        img_qualf = ImageQualification(name, img, debug, sensitivity=sensitivity)
        img_qualf.qualify(known_shape_descriptors)
        image_qualifications.append(img_qualf)

    # If descriptors found
    if not any( 0 < len(img_qualf.ordered_descriptors) for img_qualf in image_qualifications):
        print(f"Unable to qualify any of the input images. Couldn't find any correlation between input and training images.")
        sys.exit(-1)

    # Plot results
    if '-v' in argv:
        plot_confusion_matrix(image_qualifications, known_shape_descriptors)

    # Write results file
    if '-ow' not in argv and os.path.exists(dst_file_path):
        print(f"Output file '{dst_file_path}' already exists. To overwrite it add '-ow' as a flag to the running executable.")
        sys.exit(-1)

    save_result_file(dst_file_path, image_qualifications, known_shape_descriptors)

