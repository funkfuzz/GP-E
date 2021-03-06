# GP-E 
## GazePoint Estimation :eyes:

The algorithm presented in this repo uses OpenCV, Dlib and a pre-trained CNN model called [GazeML](https://github.com/swook/GazeML) to succesfully estimate a person's point of gaze in 3D space. It does so by performing the following steps: 
- Face Detection & Tracking 
- Facial Landmarks Estimation 
- Head Pose Estimation
- Gaze Direction Estimation
- Point of Gaze Estimation

Currently, the algorithm does not perform any calibration to compensate for different person's orientation regarding the camera, however this is planned to be implemented.

## Features
- Takes a single camera image as input (current default implementation works with Webcams)
- Outputs head pose orientation, facial and eye landmarks, gaze direction vectors, point of gaze (relative to camera origin)
- Visualizes of the outputs
    - The head orientation is displayed as cartesian coordinate system originating from the person's forehead
    - The facial and eye landmarks are displayed in orange 
    - The gaze direction vectors originate form the eyeball center and are coloured in cyan
    - The point of gaze is connected to the eyeball centers via a blue arrow

## Tech

- Python
- OpenCV
- Dlib
- TensorFlow

## Install dependencies

Run(with sudo appended if necessary),

```sh
 python3 setup.py install 
```   

Tensorflow should be installed separately. Inside the setup.py there is a list of all packages installed installed in my current working virtual environment. From these you can create your own [Requirements Files](https://packaging.python.org/discussions/install-requires-vs-requirements/#requirements-files) if you would prefer.

## License

MIT


