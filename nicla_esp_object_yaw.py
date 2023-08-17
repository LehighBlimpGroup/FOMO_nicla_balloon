# Edge Impulse - OpenMV Object Detection Example

from machine import I2C
from vl53l1x import VL53L1X
import sensor, image, time, os, tf, math, uos, gc, pyb
from pyb import UART


class Tracking_ROI:
    """ class Tracking_ROI:
    A square tracking window class that takes in new detection
    bounding box and adjust ROI for next detection
    """
    def __init__(self, x0=40, y0=0, max_windowsize=240, min_windowsize=96, forgetting_factor=0.5):
        self.roi = [x0, y0, max_windowsize, max_windowsize]
        self.x0 = x0
        self.y0 = y0
        self.max_windowsize = max_windowsize
        self.min_windowsize = min_windowsize
        self.ff = forgetting_factor
        self.previous_success = False

    def update(self, detection=False, x=None, y=None, w=None, h=None):
        if detection == False:
            # failed detection result in maximum tracking box
            self.roi[0] = (1 - self.ff)*self.roi[0] + self.ff*self.x0
            self.roi[1] = (1 - self.ff)*self.roi[1] + self.ff*self.y0
            self.roi[2] = (1 - self.ff)*self.roi[2] + self.ff*self.max_windowsize
            self.roi[3] = (1 - self.ff)*self.roi[3] + self.ff*self.max_windowsize
            self.previous_success = False
        else:
            # detection = True
            if x == None:
                return
            elif self.previous_success == False:
                self.previous_success == True
                self.roi[0] = (1 - self.ff)*self.roi[0] + self.ff*0.8*x
                self.roi[1] = (1 - self.ff)*self.roi[1] + self.ff*0.8*y
                self.roi[2] = (1 - self.ff)*self.roi[2] + self.ff*max(1.25*w, self.min_windowsize)
                self.roi[3] = (1 - self.ff)*self.roi[3] + self.ff*max(1.25*h, self.min_windowsize)
            else:
                self.roi[0] = (1 - self.ff)*self.roi[0] + self.ff*min([self.roi[0], 0.8*x])
                self.roi[1] = (1 - self.ff)*self.roi[1] + self.ff*min([self.roi[1], 0.8*y])
                self.roi[2] = (1 - self.ff)*self.roi[2] + self.ff*max([self.roi[2], 0.8*x + 1.25*w - self.roi[2]])
                self.roi[3] = (1 - self.ff)*self.roi[3] + self.ff*max([self.roi[3], 0.8*y + 1.25*h - self.roi[3]])
        # corner case
        if self.roi[0] + self.roi[2] > self.x0*2+self.max_windowsize:
            self.roi[2] = self.x0*2 + self.max_windowsize - self.roi[0]
        if self.roi[1] + self.roi[3] > self.y0*2+self.max_windowsize:
            self.roi[3] = self.y0*2 + self.max_windowsize - self.roi[1]
        print([int(self.roi[i]) for i in range(4)])

    def get_roi(self):
        return [int(self.roi[i]) for i in range(4)]

    def get_center(self):

        return [int(self.roi[i]) for i in range(4)]


#iBus protocol checksum
def checksum(arr, initial= 0):
    sum = initial
    for a in arr:
        sum += a
    checksum = 0xFFFF - sum
    chA = checksum >> 8
    chB = checksum & 0xFF
    return chA, chB

def sendIBUS(cx, cy, snapshot, dist):


    msg = bytearray(32)         # 16 pairs of bytes
    msg[0] = 0x20               # start bytes (the other end must synchronize to this pattern)
    msg[1] = 0x40
    cx_msg = bytearray(cx.to_bytes(2, 'little'))
    msg[2] = cx_msg[0]
    msg[3] = cx_msg[1]
    cy_msg = bytearray(cy.to_bytes(2, 'little'))
    msg[4] = cy_msg[0]
    msg[5] = cy_msg[1]
    snap_msg = bytearray(snapshot.to_bytes(2, 'little'))
    msg[6] = snap_msg[0]
    msg[7] = snap_msg[1]
    dist_msg = bytearray(int(dist).to_bytes(2, 'little'))
    msg[8] = dist_msg[0]
    msg[9] = dist_msg[1]
    chA, chB = checksum(msg[:-2], 0)
    msg[-1] = chA
    msg[-2] = chB
    uart.write(msg)             # send 32 byte message

def getCenter():
    detections = net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)], roi=tracking_ROI.get_roi())
    detected = False
    maxX = 240
    maxY = 240
    truex = 0
    truey = 0
    for i, detection_list in enumerate(detections):
        if i == 0 or len(detection_list) == 0:
            continue # background or no detections for this class
        else:
            if detected == False:
                detected = True
            for d in detection_list:
                [x, y, w, h] = d.rect()
                tracking_ROI.update(detected, x, y, w, h)
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)
                if (abs(center_x - maxX/2) < abs(truex - maxX/2) ):
                    truex = center_x
                    truey = center_y
    return truex, truey, detected

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.set_auto_exposure(False)
sensor.set_auto_whitebal(False)
sensor.__write_reg(0xfe, 0b00000000) # change to registers at page 0
print(sensor.__read_reg(0x80))
sensor.__write_reg(0x03, 0b00000000) # high bits of exposure control
sensor.__write_reg(0x04, 0b11100000) # low bits of exposure control
sensor.__write_reg(0xb0, 0b01110000) # global gain
sensor.__write_reg(0xad, 0b01110000) # R ratio
sensor.__write_reg(0xae, 0b01110000) # G ratio
sensor.__write_reg(0xaf, 0b01110100) # B ratio
# RGB gains
sensor.__write_reg(0xa3, 0b10001010) # G gain odd
sensor.__write_reg(0xa4, 0b10001010) # G gain even
sensor.__write_reg(0xa5, 0b10000000) # R gain odd
sensor.__write_reg(0xa6, 0b10000000) # R gain even
sensor.__write_reg(0xa7, 0b10010000) # B gain odd
sensor.__write_reg(0xa8, 0b10010000) # B gain even
sensor.__write_reg(0xa9, 0b10000000) # G gain odd 2
sensor.__write_reg(0xaa, 0b10000000) # G gain even 2

sensor.__write_reg(0xfe, 0b00000010) # change to registers at page 2
#sensor.__write_reg(0xd0, 0b00000000) # change global saturation,
                                      # strangely constrained by auto saturation
sensor.__write_reg(0xd1, 0b01110000) # change Cb saturation
sensor.__write_reg(0xd2, 0b01100000) # change Cr saturation
sensor.__write_reg(0xd3, 0b01000000) # luma contrast
sensor.__write_reg(0xd5, 0b00000000) # luma offset
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.75

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

gamBound = [2, 40]#1/gamma = decay factor
vlBound = [0,2000]
gamGam = 9

tof = VL53L1X(I2C(2))
dist = 1000
oldDist = tof.read()
newDist = 0
diffGam = gamGam

clock = time.clock()
# tracking ROI
tracking_ROI = Tracking_ROI(forgetting_factor = 0.25)
uart = UART("LP1", 115200, timeout_char=2000) # (TX, RX) = (P1, P0) = (PB14, PB15)

snapshot = 0
ocx = 0
ocy = 0
while(True):
    clock.tick()
    newDist = tof.read()
    diff = abs(newDist-oldDist)
    if diff > vlBound[1]:
        diff = vlBound[1]
    elif diff < vlBound[0]:
        diff = vlBound[0]
    diff = diff/vlBound[1]
    diffWeight = gamBound[0] + diff * (gamBound[1] - gamBound[0])
    diffGam = diffGam * (1 - 1/gamGam) + diffWeight*(1/gamGam)
    dist = dist*(1-1/diffGam) + newDist*(1/diffGam)
    oldDist = newDist
    print(f"Distance: {dist}mm")

    img = sensor.snapshot()#.mean(1)
    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects
    cx, cy, detected = getCenter()
    if detected and (cx != 0 and cy != 0):
        pyb.LED(2).on()
        snapshot += 1
        sendIBUS(cx, cy, snapshot, dist)
        if snapshot > 1000:
            snapshot = 1
        ocx = cx
        ocy = cy
    else:
        sendIBUS(ocx, ocy, snapshot, dist)

        pyb.LED(2).off()

    # if no detection, resume initial roi, otherwise do nothing
    tracking_ROI.update(detected)

    # Draw the rectangle on the image
    rect_color = (255, 0, 0)  # RGB color (red)
    img.draw_rectangle(tracking_ROI.get_roi(), color=rect_color)
    tracking_ROI


    print(clock.fps(), "fps", end="\n\n")






