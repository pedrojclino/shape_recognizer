# Shape Recognizer

## Description
Finds a group of shape locations on input images, by using the trained model data.
The algorithm uses classical approaches and OpenCV to get the job done.

- **Motivation**: Why did you create this?
- **Problem Solved**: What need does this address?
- **Key Features**: What makes your project stand out?
- **What You Learned**: Any key technical or conceptual takeaways.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Help Commands](#help-commands)
- [Examples](#examples)
- [Media](#media)
- [Credits](#credits)
- [License](#license)

## Compilation
I'd recommend using the pre-compiled runnables inside \bin: "shape_recognizer.exe" and "train_shapes.exe". 
To compile the code yourself directly from the repository directory, you may run:

py -m PyInstaller --onefile --add-data ../data/model.json:. --icon=../res/icons/app_icon.ico --distpath ../bin --workpath ../build --specpath ../build --name shape_recoginzer main.py

"train_shapes.exe"
py -m PyInstaller --onefile --icon=../res/icons/app_icon.ico --distpath ../bin --workpath ../build --specpath ../build --name train_shapes train.py

Note: some python packages may have to be installed for the installation to be successful.

## Usage
"shape_recognizer.exe" will ...
Run the precompiled binaries and test it yourself run:
.\bin\shape_recognizer.exe

"train_shapes.exe" will ...
To run the training on a group of good shape samples run:
.\bin\train_shapes.exe

## Help Commands
- Option flags for "shape_recognizer.exe":
Use '-p' followed by the path to a folder containing subfolders each with a group of references images to compose the model.
Add '-r' to overwrite the previous model.json file.
Add '-d' to debug the shapes being trained.
Use '-h' for help.

- Options flags "train_shapes.exe":
Use '-p' followed by the path to a folder containing subfolders each with a group of references images to compose the model.
Add '-r' to overwrite the previous model.json file.
Add '-d' to debug the shapes being trained.
Use '-h' for help.

## License
MIT License. Use the code however you please, it's free!

## Credits
Pedro Lino