# Takes in an argument for which camera to change.
# Set the values you want
brightness=0
contrast=32
gamma=100
gain=0
sharpness=3
exposure_time=166

x_res=640
y_res=480

# Actually setting stuff
v4l2-ctl -d $1 -c brightness=$brightness
echo "Set cam$1 brightness to $brightness"
v4l2-ctl -d $1 -c contrast=$contrast
echo "Set cam$1 contrast to $contrast"
v4l2-ctl -d $1 -c gamma=$gamma
echo "Set cam$1 gamma to $gamma"
v4l2-ctl -d $1 -c gain=$gain
echo "Set cam$1 gain to $gain"
v4l2-ctl -d $1 -c sharpness=$sharpness
echo "Set cam$1 sharpness to $sharpness"

v4l2-ctl -d $1 -c auto_exposure=3
echo "Set cam$1 exposure to Auto"
v4l2-ctl -d $1 -c exposure_time_absolute=$exposure_time
echo "Set cam$1 exposure time to $exposure_time"