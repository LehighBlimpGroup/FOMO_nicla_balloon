# FOMO_nicla_balloon
A NN-based purple/green balloon detection demo

### ei_object_detection.py 
* run this script to inspect the neural network-based detection in a real environment with a real camera

### niclavisionsettings.py
* run this script to collect training data at the resolution of QVGA, in the format of a MJPEG video. Extract the individual frames using ffmpeg command `ffmpeg -i mjpegvideo.avi -vcodec copy frame%d.jpg`.
* Change the variable on line 20 `num_frames = 100` to increase/decrease the number of frames to collect.
