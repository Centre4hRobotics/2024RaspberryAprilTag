# Set the values you want
brightness=-64
contrast=64
gamma=72
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

v4l2-ctl -d 0 -c auto_exposure=1
echo "Set cam0 exposure to Manual"
v4l2-ctl -d 0 -c exposure_time_absolute=$exposure_time
echo "Set cam0 exposure time to $exposure_time"

v4l2-ctl -d 0 --set-fmt-video=width=$x_res,height=$y_res
echo "Set cam0 resolution to $x_res by $y_res"

v4l2-ctl -d 2 -c brightness=$brightness
v4l2-ctl -d 2 -c contrast=$contrast
v4l2-ctl -d 2 -c gamma=$gamma
v4l2-ctl -d 2 -c gain=$gain
v4l2-ctl -d 2 -c sharpness=$sharpness

v4l2-ctl -d 2 -c auto_exposure=1
v4l2-ctl -d 2 -c exposure_time_absolute=$exposure_time
