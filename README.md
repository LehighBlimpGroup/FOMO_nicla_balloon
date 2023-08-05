# FOMO_nicla_balloon
A NN-based purple/green balloon detection demo based on the FOMO (faster objects, more objects) architecture from [Edge Impulse](https://github.com/edgeimpulse). The NN is composed of 1 MobileNetV2 block and 2 CNN layers. The input image size to the MobileNet is set to 96 x 96. The network is successfully deployed on a Nicla Vision on QVGA resolution (shrunk to 96x96 before feeding to the NN) with a performance of ~15fps. 

Here is a [demo video](https://github.com/Jarvis-X/FOMO_nicla_balloon/blob/main/highbay_sunset_test.mp4).


### ei_object_detection.py 
* run this script to inspect the neural network-based detection in a real environment with a real camera

### niclavisionsettings.py
* run this script to collect training data at the resolution of QVGA, in the format of a MJPEG video. Extract the individual frames using ffmpeg command `ffmpeg -i mjpegvideo.avi -vcodec copy frame%d.jpg`.
* Change the variable on line 20 `num_frames = 100` to increase/decrease the number of frames to collect.
