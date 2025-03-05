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
v4l2-ctl -d 0 -c brightness=$brightness
echo "Set cam0 brightness to $brightness"
v4l2-ctl -d 0 -c contrast=$contrast
echo "Set cam0 contrast to $contrast"
v4l2-ctl -d 0 -c gamma=$gamma
echo "Set cam0 gamma to $gamma"
v4l2-ctl -d 0 -c gain=$gain
echo "Set cam0 gain to $gain"
v4l2-ctl -d 0 -c sharpness=$sharpness
echo "Set cam0 sharpness to $sharpness"

v4l2-ctl -d 0 -c auto_exposure=3
echo "Set cam0 exposure to Auto"
v4l2-ctl -d 0 -c exposure_time_absolute=$exposure_time
echo "Set cam0 exposure time to $exposure_time"

v4l2-ctl -d 2 -c brightness=$brightness
echo "Set cam2 brightness to $brightness"
v4l2-ctl -d 2 -c contrast=$contrast
echo "Set cam2 contrast to $contrast"
v4l2-ctl -d 2 -c gamma=$gamma
echo "Set cam2 gamma to $gamma"
v4l2-ctl -d 2 -c gain=$gain
echo "Set cam2 gain to $gain"
v4l2-ctl -d 2 -c sharpness=$sharpness
echo "Set cam2 sharpness to $sharpness"

v4l2-ctl -d 2 -c auto_exposure=3
echo "Set cam2 exposure to Auto"
v4l2-ctl -d 2 -c exposure_time_absolute=$exposure_time
echo "Set cam2 exposure time to $exposure_time"
