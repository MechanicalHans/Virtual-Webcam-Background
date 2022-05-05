Create a virtual camera by overlaying the foreground objects of a real camera onto a static background image using selfie segmentation powered by Google's [MediaPipe](https://google.github.io/mediapipe/solutions/selfie_segmentation).

This program uses the height and width of the background image to determine the dimensions to use for the real and virtual cameras. You should therefore resize the background image to your desired dimensions.

By default this program uses the "first" physical and virtual cameras. These cameras are often not the intended cameras. If this is an issue for you see the program help for how to manually set the physical and virtual devices.

# Requirements

You will need [python](https://www.python.org/) and the following dependencies:

- [NumPy](https://pypi.org/project/numpy/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [pyvirtualcam](https://pypi.org/project/pyvirtualcam/)
- [MediaPipe](https://pypi.org/project/mediapipe/)

Relatively recent versions of these dependencies should work fine. If not try updating these dependencies to the newest versions. If the issue still persists please file an issue.

Additionally, you will need to install a virtual camera for [pyvirtualcam](https://pypi.org/project/pyvirtualcam/) to connect to. See its documentation for a list of supported virtual cameras.