import cv2
import argparse
import time
from PIL import Image
from screeninfo import get_monitors

from dreamHelper import *



def take_photo(vc):
    t0 = time.time()
    while True:
        rval, frame = vc.read()
        cv2.imshow("preview", frame)
        if time.time() - t0 >= 0.01:
            print("taken photo")
            break
    return frame




def deep_dream(deepdream, frame, i, resize_factor, size, octave_bool, steps, step_size, octave_scale, octaves,
               zoom_factor, take_photos, tile_size, photo_interval, redream, blend):
    
    size_small = [int(size[0]*resize_factor), int(size[1]*resize_factor)]
    original_img = np.array(frame)
    original_img = cv2.resize(original_img, dsize=size_small)
    if octave_bool:
        dream_img = run_deep_dream_with_octaves(img=original_img, dd_model=deepdream,
                                                steps_per_octave=steps, step_size=step_size,
                                                octave_scale=octave_scale, octaves=octaves, tile_size=tile_size)

    else:
        
        dream_img = run_deep_dream_simple(img=original_img, dd_model=deepdream, steps=steps,
                                        step_size=step_size)

    print("updated")

    dream_array = np.array(dream_img)

    if dream_array.shape != original_img.shape:
        original_img = cv2.resize(original_img, [dream_array.shape[1], dream_array.shape[0]])

    dream_array = cv2.addWeighted(dream_array, blend, original_img, 1-blend, gamma=0)

    original_img = crop_pic(dream_array, zoom_factor)
    original_img = cv2.resize(original_img, dsize=size, interpolation=cv2.INTER_CUBIC)

    if redream:
        if octave_bool:
            dream_img = run_deep_dream_with_octaves(img=original_img, dd_model=deepdream,
                                                    steps_per_octave=steps, step_size=step_size,
                                                    octave_scale=octave_scale, octaves=octaves, tile_size=tile_size)

        else:
            dream_img = run_deep_dream_simple(img=original_img, dd_model=deepdream, steps=steps,
                                              step_size=step_size)
        original_img = np.array(dream_img)

    if i % photo_interval == 0 and take_photos:
        dream_as_img = tf.keras.preprocessing.image.array_to_img(original_img)
        dream_as_img.save('intermediate' + str(i) +'.png')

    return original_img
        


def project(image):
    cv2.imshow('image', image)
    cv2.waitKey(1)


def complete_run(vc, deepdream, i, size, octaves, steps, step_size, octave_scale, octave_range, zoom_factor, resize_factor,
                 take_photos, tile_size, frame, photo_interval, redream, no_camera, blend):
    start = time.time()

    if not no_camera:
        image = take_photo(vc=vc)
    else:
        image = frame
    image = deep_dream(deepdream=deepdream, frame=image, i=i, size=size, resize_factor=resize_factor,
                       octave_bool=octaves, steps=steps, step_size=step_size, octave_scale=octave_scale,
                       octaves=octave_range, zoom_factor=zoom_factor, take_photos=take_photos,
                       tile_size=tile_size, photo_interval=photo_interval, redream=redream, blend=blend)
    project(image=image)


    print(time.time() - start)
    return image



def main(octaves, frames, names, steps, step_size, octave_scale, octave_range, zoom_factor, resize_factor, take_photos,
         tile_size, no_camera, photo_interval, redream, blend, extension):

    # set up dreaming model
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    layers = [base_model.get_layer(name).output for name in names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    if octaves: dreamer = TiledGradients
    else: dreamer = DeepDream
    deepdream = dreamer(dream_model)

    # get dimensions of second monitor
    display2_x = get_monitors()[0].width
    display2_y = get_monitors()[0].height

    # open image and get size
    if no_camera:
        image = Image.open('image' + extension)
        img_shape = image.size
        height = img_shape[0]
        width = img_shape[1]
        vc = None
    # open camera and get camera dimensions
    else:
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        if vc.isOpened():
            width = int(vc.get(3))
            height = int(vc.get(4))
            img_shape = (height, width)
        else:
            raise Exception("Camera not open")
        image = ""

    # open window for projection
    aspect_ratio = height / width
    cv2.namedWindow('image', flags=cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow('image', display2_x, 0)
    cv2.resizeWindow('image', int(display2_y / aspect_ratio), display2_y)
    black_img = np.zeros((height, width, 3), dtype='uint8')
    cv2.imshow('image', black_img)
    cv2.waitKey(1)

    # main loops
    if frames == -1:
        i = 0
        while True:
            if cv2.waitKey(1) == 27:
                print("break")
                break
            image = complete_run(vc=vc, deepdream=deepdream, i=i, octaves=octaves, steps=steps, step_size=step_size,
                                 octave_scale=octave_scale, octave_range=octave_range, zoom_factor=zoom_factor,
                                 resize_factor=resize_factor, size=img_shape, take_photos=take_photos,
                                 tile_size=tile_size, frame=image, photo_interval=photo_interval, redream=redream,
                                 no_camera=no_camera, blend=blend)
            if cv2.waitKey(1) == 13:
                cv2.imwrite("out" + str(i) + ".jpg", image)
            i += 1
    else:
        for i in range(frames):
            image = complete_run(vc=vc, deepdream=deepdream, i=i, octaves=octaves, steps=steps, step_size=step_size,
                                 octave_scale=octave_scale, octave_range=octave_range, zoom_factor=zoom_factor,
                                 resize_factor=resize_factor, size=img_shape, take_photos=take_photos,
                                 tile_size=tile_size, frame=image, photo_interval=photo_interval, redream=redream,
                                 no_camera=no_camera, blend=blend)

    # close camera and other windows
    if not no_camera:
        vc.release()
    cv2.destroyAllWindows()

    cv2.imwrite("out.jpg", image)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--octave_bool', help="whether to use octaves", type=bool, default=True)
    parser.add_argument('--frames', help="number of pictures", type=int, default=5)
    parser.add_argument('--layers', help="which layers to include", nargs="+", default=['activation', 'mixed5', 'conv2d_28'])
    parser.add_argument('--steps', help="number of steps", type=int, default=8)
    parser.add_argument('--step_size', help="step size", type=float, default=0.028)
    parser.add_argument('--octave_scale', help="octave scale", type=float, default=1.001)
    parser.add_argument('--octave_range_1', help="first value for octave range", type=int, default=543)
    parser.add_argument('--octave_range_2', help="second value for octave range", type=int, default=548)
    parser.add_argument('--zoom_factor', help="zoom per frame", type=float, default=.985)
    parser.add_argument('--resize_factor', help="how much to decrease image quality", type=float, default=.4)
    parser.add_argument('--take_photos', help="whether to save intermediate photos", type=bool, default=True)
    parser.add_argument('--photo_interval', help="after how many photos to save", type=int, default=1)
    parser.add_argument('--tile_size', help="tile size for octave dream", type=int, default=512)
    parser.add_argument('--no_camera', help="if true, dreams on last image", type=bool, default=True)
    parser.add_argument('--re_dream', help="whether to redream on the full quality image", type=bool, default=True)
    parser.add_argument('--blend', help="how much to blend last image, 1=only current image, 0=only last", type=float, default=0.8)
    parser.add_argument('--extension', help="if no_camera, file extension of image to open", type=str, default=".jpg")

    args=parser.parse_args()

    main(octaves=args.octave_bool, frames=args.frames, names=args.layers, steps=args.steps, step_size=args.step_size,
         octave_scale=args.octave_scale, octave_range=range(args.octave_range_1, args.octave_range_2),
         zoom_factor=args.zoom_factor, resize_factor=args.resize_factor, take_photos=args.take_photos,
         tile_size=args.tile_size, no_camera=args.no_camera, photo_interval=args.photo_interval, redream=args.re_dream,
         blend=args.blend, extension=args.extension)