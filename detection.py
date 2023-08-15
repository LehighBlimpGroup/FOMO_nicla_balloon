# detection - By: Bingxu - Fri Aug 4 2023
import sensor, image, time, os, tf, math, uos, gc, network, rpc, omv, mjpeg, pyb, random
from pyb import UART

#Blink LED indicating resetting
def setting_up():
    print("setting up the device...")
    pyb.LED(2).on()
    time.sleep_ms(1000)
    pyb.LED(2).off()
    time.sleep_ms(500)
    pyb.LED(2).on()
    time.sleep_ms(1000)
    pyb.LED(2).off()
#iBus protocol checksum
def checksum(arr, initial= 0):
    sum = initial
    for a in arr:
        sum += a
    checksum = 0xFFFF - sum
    chA = checksum >> 8
    chB = checksum & 0xFF
    return chA, chB

#Initialize UART on OpenMV H7
uart = UART("LP1", 115200, timeout_char=2000) # (TX, RX) = (P1, P0) = (PB14, PB15)

#Initialize clock
clock = time.clock()

#Sensor setup
setting_up()
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((120, 120))       # Set 240x240 window.
sensor.set_auto_exposure(False)
sensor.set_auto_whitebal(False)
sensor.__write_reg(0xfe, 0b00000000) # change to registers at page 0
sensor.__write_reg(0x03, 0b00000001) # high bits of exposure control
sensor.__write_reg(0x04, 0b10000000) # low bits of exposure control
sensor.__write_reg(0xb0, 0b01100000) # global gain
sensor.__write_reg(0xad, 0b10010000) # R
sensor.__write_reg(0xae, 0b10100100) # G
sensor.__write_reg(0xaf, 0b10101000) # B

sensor.__write_reg(0xfe, 0b00000010) # change to registers at page 2
#sensor.__write_reg(0xd0, 0b00000000) # change global saturation,
                                      # strangely constrained by auto saturation
sensor.__write_reg(0xd1, 0b10110000) # change Cb saturation
sensor.__write_reg(0xd2, 0b10110000) # change Cr saturation
sensor.__write_reg(0xd3, 0b01000100) # luma contrast

sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.92

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

red_led = pyb.LED(1)
green_led = pyb.LED(2)
blue_led = pyb.LED(3)
#Loop
while(True):
    time.clock().tick()
    img = sensor.snapshot().crop(x_scale=0.8, y_scale=0.8)#.mean(1)

    msg = bytearray(32)         # 16 pairs of bytes
    msg[0] = 0x20               # start bytes (the other end must synchronize to this pattern)
    msg[1] = 0x40

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) != 0):
            pyb.LED(1).on()
            for d in detection_list:
                [x, y, w, h] = d.rect()
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                print('x %d\ty %d' % (center_x, center_y))
                img.draw_circle((center_x, center_y, 4), color=colors[i], thickness=2)
                center_x = math.floor((x + (w / 2)))
                center_y = math.floor((y + (h / 2)))
                cx_msg = bytearray(center_x.to_bytes(2, 'little'))
                msg[2] = cx_msg[0]
                msg[3] = cx_msg[1]
                cy_msg = bytearray(center_y.to_bytes(2, 'little'))
                msg[4] = cy_msg[0]
                msg[5] = cy_msg[1]
                time.sleep_ms(20)
        else:
            msg[2] = 0x0
            msg[3] = 0x0
            msg[4] = 0x0
            msg[5] = 0x0
            red_led.off()

    #iBus protocol checksum
    chA, chB = checksum(msg[:-2], 0)
    msg[-1] = chA
    msg[-2] = chB
    uart.write(msg)             # send 32 byte message
    print(msg)


