import cv2
import argparse
import time
from screeninfo import get_monitors

from dreamHelper import *


def take_photo(vc):
    t0 = time.time()
    while True:
        rval, frame = vc.read()
        cv2.imshow("preview", frame)
        if time.time() - t0 >= 0.001:
            break
    return frame


def deep_dream(deepdream, frame,  octave_bool, steps, step_size, octave_scale, octaves, zoom_factor):

    original_img_png = frame
    original_img = np.array(original_img_png)

    if octave_bool:
        dream_img = run_deep_dream_with_octaves(img=original_img, dd_model=deepdream,
                                                steps_per_octave=steps, step_size=step_size,
                                                octave_scale=octave_scale, octaves=octaves)
    else:
        dream_img = run_deep_dream_simple(img=original_img, dd_model=deepdream, steps=steps,
                                        step_size=step_size)

    dream_array = np.array(dream_img)
    original_img = crop_pic(dream_array, zoom_factor)
    print("updated")

    #dream_as_img = tf.keras.preprocessing.image.array_to_img(dream_img)
    #dream_as_img.save('leafy_zoom' + str(i) +'.png')
    
    return original_img


def project(image):
    cv2.imshow('image', image)
    cv2.waitKey(1)

def complete_run(vc, deepdream, octaves, steps, step_size, octave_scale, octave_range, zoom_factor):
    start = time.time()

    image = take_photo(vc=vc)
    image = deep_dream(deepdream=deepdream, frame=image, octave_bool=octaves, steps=steps, step_size=step_size,
                       octave_scale=octave_scale, octaves=octave_range, zoom_factor=zoom_factor)
    project(image=image)
    print(image)

    print(time.time() - start)


def main(octaves, frames, names, steps, step_size, octave_scale,
         octave_range, zoom_factor):

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    layers = [base_model.get_layer(name).output for name in names]
    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    if octaves: dreamer = TiledGradients
    else: dreamer = DeepDream
    deepdream = dreamer(dream_model)

    display2_x = get_monitors()[0].width
    display2_y = get_monitors()[0].height

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        width = vc.get(3)
        height = vc.get(4)
        img_shape = (height, width)
    else:
        raise Exception("Camera not open")

    aspect_ratio = img_shape[0] / img_shape[1]
    cv2.namedWindow('image', flags=cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('image', display2_x, 0)
    cv2.resizeWindow('image', int(display2_y / aspect_ratio), display2_y)
    black_img = np.zeros((height, width, 3), dtype='uint8')
    cv2.imshow('image', black_img)
    cv2.waitKey(0)

    if frames == -1:
        while True:
            complete_run(vc=vc, deepdream=deepdream, octaves=octaves, steps=steps, step_size=step_size,
                         octave_scale=octave_scale, octave_range=octave_range, zoom_factor=zoom_factor)
    else:
        for i in range(frames):
            complete_run(vc=vc, deepdream=deepdream, octaves=octaves, steps=steps, step_size=step_size,
                         octave_scale=octave_scale, octave_range=octave_range, zoom_factor=zoom_factor)

    vc.release()
    cv2.destroyAllWindows()
    try:
        out_img = tf.keras.preprocessing.image.array_to_img(image)
        cv2.imwrite("out.jpg", out_img)
    except ValueError:
        print("Output image could not be written to file")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--octave_bool', help="whether to use octaves", type=bool, default=True)
    parser.add_argument('--frames', help="number of pictures", type=int, default=-1)
    parser.add_argument('--layers', help="which layers to include", nargs="+", default=['conv2d_45', 'mixed8'])
    parser.add_argument('--steps', help="number of steps", type=int, default=1)
    parser.add_argument('--step_size', help="step size", type=float, default=0.01)
    parser.add_argument('--octave_scale', help="octave scale", type=float, default=1.3)
    parser.add_argument('--octave_range_1', help="first value for octave range", type=int, default=-2)
    parser.add_argument('--octave_range_2', help="second value for octave range", type=int, default=3)
    parser.add_argument('--zoom_factor', help="zoom per step", type=float, default=0.95)

    args=parser.parse_args()

    main(octaves=args.octave_bool, frames=args.frames, names=args.layers, steps=args.steps, step_size=args.step_size,
         octave_scale=args.octave_scale, octave_range=range(args.octave_range_1, args.octave_range_2),
         zoom_factor=args.zoom_factor)
