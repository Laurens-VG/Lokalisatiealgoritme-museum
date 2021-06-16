# Project Computervisie Groep 5
Project computervisie van groep 5: Lokalisatie in het MSK

Universiteit Gent \
Faculteit Ingenieurswetenschappen en architectuur \
Opleiding Industriële Wetenschappen Elektronica-ICT \
Academiejaar 2019-2020

Auteurs: Maxime Carella, Reiner De Smet, Laurens Van Goethem, Elias Vanhoutte\
Project onder begeleiding van: Sanne Roegiers, David Van Hamme

## Folder: RESULTS
excel files from final results

## Folder: code
- **floorplan**: Folder with info for localization
- **test_output**: For testing purposes
- **test_output60**: For testing purposes
- **test_output80**:For testing purposes
- **test_output_sharpness15**: For testing purposes
- **test_output_sharpness25**: For testing purposes
- **database_creation.py**: Creation of the database
- **main_task2.py**: Not used
- **main_task3.py**: Not used
- **calibration.py**: Calibrates the camera, determines calibration parameters
- **calibration_data**: Camera parameters for undistorting
- **undistort.py**: Create undistorted videos
- **config.py**: Configures the project, sets path to source files and parameters
- **main_final.py**: Main file to run
- **file_tools.py**: File for creating CSV files, not used
- **pickle_tools.py**: For storing and loading discriptors for pickle files
- **warping.py**: warps image to correct aspects ratio 
- **localization.py**: Visual localization of floorplan to show user position and path
- **painting_detection.py**: Extracting paintings from videoframes


## Gebruik

Het project gebruikt python 3.7 en python 3.8. \
De packages (zie versie controle) kunnen geïnstalleerd worden met pip:

    pip install -r requirements.txt
    
Download database via https://ugentbe-my.sharepoint.com/:u:/g/personal/elias_vanhoutte_ugent_be/Edt6AcNC3cpLpXNx3Vu_EvwBtYdcsryTDsyPfv9JdAs7sg?e=s5qcZ2 \
Initialiseer config.py.\
Run undistort.py.\
Run main_final.py.\
First run descriptor files will be stored to path specified in the config.py file.\
Second run, this would be used for a speed boost.

## Versie controle

OS: Windows 10

IDE: JetBrains Pycharm 2019.2.2 (Professional Edition)
IDE: Anaconda Spyder 3.x.x (Elias)

Python 3.7.1\
Python 3.8

Packages
- numpy
- opencv-python
- matplotlib
- tools
- scipy
- scikit-learn
- scikit-image


