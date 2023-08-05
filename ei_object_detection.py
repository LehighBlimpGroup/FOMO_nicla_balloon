# Edge Impulse - OpenMV Object Detection Example

import sensor, image, time, os, tf, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.set_auto_exposure(False)
sensor.set_auto_whitebal(False)
sensor.__write_reg(0xfe, 0b00000000) # change to registers at page 0
sensor.__write_reg(0x03, 0b00000000) # high bits of exposure control
sensor.__write_reg(0x04, 0b01100000) # low bits of exposure control
sensor.__write_reg(0xb0, 0b11010000) # global gain
sensor.__write_reg(0xad, 0b10001100) # R
sensor.__write_reg(0xae, 0b10000000) # G
sensor.__write_reg(0xaf, 0b10001110) # B

sensor.__write_reg(0xfe, 0b00000010) # change to registers at page 2
#sensor.__write_reg(0xd0, 0b00000000) # change global saturation,
                                      # strangely constrained by auto saturation
sensor.__write_reg(0xd1, 0b01101100) # change Cb saturation
sensor.__write_reg(0xd2, 0b01001000) # change Cr saturation
sensor.__write_reg(0xd3, 0b01001000) # luma contrast

sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (0,   0,   0),
    (0, 255,   0),
    (128, 0,   128),
]

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot().crop(x_scale=0.4, y_scale=0.4)

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?

        print("********** %s **********" % labels[i])
        for d in detection_list:
            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 4), color=colors[i], thickness=2)

    print(clock.fps(), "fps", end="\n\n")
