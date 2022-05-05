#!/usr/bin/env python3

import argparse as ap # https://docs.python.org/3/library/argparse.html
import io # https://docs.python.org/3/library/io.html
import cv2 as cv # https://docs.opencv.org/4.5.5/
import pyvirtualcam as vc # https://letmaik.github.io/pyvirtualcam/
import mediapipe as mp # https://google.github.io/mediapipe/solutions/selfie_segmentation#python-solution-api
import numpy as np # https://numpy.org/doc/stable/

def main():
    args = create_argument_parser().parse_args()
    background = read_image(args.background)
    height, width, _ = background.shape
    # Don't use the `with` syntax here because we're at the top level and any exceptions should end the program. Object
    # deconstructors should still run on program termination and any acquired resources should still be freed.
    segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=args.model)
    transformer = create_frame_overlayer(background, segmentor, args.threshold)
    physical = create_handle_to_physical(args.physical, width, height, args.frame_rate, args.codec)
    virtual = vc.Camera(width, height, args.frame_rate, device=args.virtual, print_fps=args.loud)
    if args.loud: print(
        f"Camera dimensions {width}*{height}@{args.frame_rate} using codec {args.codec}",
        f"Model choice {args.model} using threshold {args.threshold}%",
        sep='\n'
    )
    pipe_through(physical, virtual, transformer)

def create_argument_parser():
    parser = ap.ArgumentParser(description="""
Create a virtual camera by overlaying the foreground objects of a real camera onto a static background image.
        """,
    )
    parser.add_argument('background', type=ap.FileType(mode='rb'), help="background image path")
    parser.add_argument('frame_rate', type=int, help="virtual camera target frame rate")
    parser.add_argument('--physical', type=str, help="physical camera path")
    parser.add_argument('--virtual', type=str, help="virtual camera path")
    parser.add_argument('--codec', type=str, default='MJPG', help="physical camera fourcc codec")
    parser.add_argument('--model', type=int, choices=[0, 1], default=1, help="MediaPipe model kind")
    parser.add_argument('--threshold', type=float, default=80.0, help="MediaPipe percentage confidence threshold")
    parser.add_argument('--silent', dest='loud', action='store_false', help="disable output")
    return parser

def create_handle_to_physical(path, width, height, rate, codec):
    camera = cv.VideoCapture()
    identifier = 0 if path is None else path
    if not camera.open(identifier): raise ValueError("could not open camera")
    try: fourcc = cv.VideoWriter_fourcc(*codec)
    except TypeError: raise ValueError("codec is not a four character code")
    if not camera.set(cv.CAP_PROP_FOURCC, fourcc): raise ValueError("could not set codec to: {codec}")
    if not camera.set(cv.CAP_PROP_FRAME_WIDTH, width): raise ValueError("could not set frame width to: {width}")
    if not camera.set(cv.CAP_PROP_FRAME_HEIGHT, height): raise ValueError("could not set frame height to: {height}")
    if not camera.set(cv.CAP_PROP_FPS, rate): raise ValueError("could not set frame rate to: {rate}")
    return camera

def create_frame_overlayer(background, segmentor, threshold):
    if not 0 <= threshold < 100: raise ValueError("threshold is not between 0 and 100")
    threshold /= 100 # threshold is given as a percentage but mediapipe needs it in the interval [0, 1]
    def foreground_overlayer(frame):
        frame.flags.writeable = False # this is supposed to improve performance according to the MediaPipe docs
        results = segmentor.process(frame)
        mask = results.segmentation_mask > threshold
        mask = np.stack((mask,) * 3, axis=-1) # MediaPipe works on a per pixel basic but we need a per subpixel mask
        return np.where(mask, frame, background)
    return foreground_overlayer

def pipe_through(source, sink, transformer):
    while True: 
        # read frame
        success, frame = source.read()
        if not success: continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # transform frame
        frame = transformer(frame)
        # write frame
        sink.send(frame)
        sink.sleep_until_next_frame()

# avoid using `cv.imread` because it clobbers the information about why reading the image file failed
def read_image(file: io.BufferedIOBase) -> np.ndarray:
    contents = np.frombuffer(file.read(), dtype=np.uint8) 
    decoded = cv.imdecode(contents, flags=cv.IMREAD_COLOR)
    if decoded is None: ValueError("could not decode background image file")
    return decoded

if __name__ == '__main__':
    try: main()
    # silence output caused by receiving a termination signal because that is the intended way to end the program
    except KeyboardInterrupt: pass
