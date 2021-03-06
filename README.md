# ๐AIFFEL ๋์  1๊ธฐ ์์จ์ฃผํ ํ๋ก์ ํธ๐
์์จ์ฃผํ, ํ๋๋ก๋ด ํ๋ซํผ์ ์ ๊ณตํ๋ ๊ธฐ์ ์๊ณ  ์ฝ๋ฆฌ์์ ํ์ํ ํ๋ก์ ํธ [์๊ณ ์ฝ๋ฆฌ์ ํํ์ด์ง](https://wego-robotics.com/)

## 1. ํ๋ช: We-Go
## 2. ์ผ์ : 2021.05.10 ~ 2021.06.18 (์ฝ 6์ฃผ)
## 3. ํ์: ์์ฐฝ์(ํ์ฅ), ๊น๊ฐํ, ์์ง์ , ์์ํ, ๋ฌธ์ฌ์ค
## 4. ๋ชฉํ: ์ค๋ด ์์จ์ฃผํ ๋ชจ๋ฐ์ผ ๋ก๋ด์ด ํน์  ์ธ๋ฌผ์ Tracking ํ  ์ ์๋ ๊ธฐ๋ฅ ๊ตฌํ
## 5. ์ญํ 
|์ญํ |main|sub|
|---|---|---|
|Tracking(Siam)|๊น๊ฐํ|-|
|Tracking(DeepSORT)|์์ฐฝ์|-|
|depth camera|์์ํ|๊น๊ฐํ|
|ROS|๋ฌธ์ฌ์ค|์์ฐฝ์|
|H/W|๋ฌธ์ฌ์ค|์์ง์ |
|Recording|์์ง์ |-|

### [ํ๋ก์ ํธ ์งํ Notion](https://www.notion.so/We-Go-ed512708c2f14177a53e4f5c95d918a9)

# Code ์ฌ์ฉ ๋ฐฉ๋ฒ
## ์ค์น(Installation)
1. ROS  
2. scout-mini  
3. yolov4-deepsort  

## 1. ROS ์ค์น & workspace init
1) [ROS ์ค์น ๋งํฌ](http://wiki.ros.org/melodic/Installation/Ubuntu)๋ก ์ด๋  
2) ROS Melodic(Ubuntu 18.04 ํธํ ๋ฒ์ ) ์ค์น  
3) update๊น์ง ๋ง์น๊ณ  desktop-full ์คํ  
`$ sudo apt install ros-melodic-desktop-full`  
4) 1.6.1๊น์ง ์งํ  
5) ์ค์น ํ ํฐ๋ฏธ๋์์ `roscore` ์คํ์ผ๋ก ์ ์์ ์ผ๋ก ์ค์น๋์๋์ง ํ์ธ  

![roscore](image/roscore.png)  

์ฐธ๊ณ ) ROS Melodic์์ Python3๋ฅผ ์ฌ์ฉํ๊ธฐ ์ํด์๋ ์๋ ๋ช๋ น์ด ์๋ ฅ ํ์  
`$ sudo apt-get install python3-catkin-pkg-modules`  
`$ sudo apt-get install python3-rospkg-modules`

6) ํฐ๋ฏธ๋ ์ฐฝ์์ ์๋์ ๊ฐ์ด ์์ ๊ณต๊ฐ(ํด๋)๋ฅผ ์์ฑํ๋ค. (catkin_ws ์ด์ธ์ ๋ค๋ฅธ ํด๋ ์ด๋ฆ์ ํด๋ ์๊ด์๋ค.)  
`$ cd ~ && mkdir -p catkin_ws/src`  
`$ cd ~/catkin_ws/src`  
7) workspace init ์ค์  
`$ catkin_init_workspace`  

## 2. scout-mini, yolov4-deepsort ์ค์น  
1) ์ ๋ด์ฉ๊ณผ ์ด์ด์ง. scout-mini github code๋ฅผ cloneํด์ผ ํ๋ค. ROS workspace initํ ์ํ์์ ๋ฐ๋ก ์งํํ๋ค.  
`$ git clone https://github.com/We-Go-Autonomous-driving/main2_one_person.git`  
2) ์๋ก์ด ํจํค์ง(ํด๋)๋ฅผ ์ค์นํ๋ฉด catkin_make๋ฅผ ํด์ค์ผ ํ๋ค.(์์ ํด๋์์ ํด์ผํจ)  
`$ cd .. && catkin_make`  

์ฐธ๊ณ ) Python ํ์ผ์ ์๋ก ์์ฑํ ํ์๋ ํด๋น ํ์ผ์ ๊ถํ ์ค์ ์ด ํ์ํ๋ค.  
`$ sudo chmod +x (ํ์ผ์ด๋ฆ)`  
๋๋ ๋ชจ๋  ํ์ผ์ ๋ํด์ ํ ๋ฒ์ ํ  ๋๋ ์๋์ ๊ฐ์ ๋ช๋ น์ด ์ฌ์ฉ  
`$ sudo chmod +x ./*`  

์ฌ๊ธฐ๊น์ง ํ๋ฉด scout-mini๋ฅผ ์ ์ดํ  ์ ์๋ ๋จ๊ณ๊ฐ ๋๋ค.  

yolov4-deepsort๋ฅผ ์ฌ์ฉํ๊ธฐ ์ํด์๋ [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) ๋ฅผ ๋ค์ด๋ฐ๊ฑฐ๋ ํน์ [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)๋ฅผ ๋ค์ด๋ฐ์์ผ ํ๋ค. ๊ทธ๋ฆฌ๊ณ  `weights`ํ์ผ์ `scout_bringup/data`๊ฒฝ๋ก์ ๋ฃ์ด์ค์ผ ํจ.

๋ํ ์๋ ๋ช๋ น์ด๋ฅผ ์คํํด์ darknet weights๋ฅผ Tensorflow model์ ์ฌ์ฉํ  ์ ์๊ฒ convertํด์ผ ํจ.  
`$ python save_model.py --model yolov4` (yolov4.weights ์ฌ์ฉ)  
`$ python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny` (yolov4-tiny.weights ์ฌ์ฉ)  

yolov4-deepsort์ ๋ํด ๋ ์์ธํ ์๊ณ  ์ถ๋ค๋ฉด [์ฌ๊ธฐ](https://github.com/theAIGuysCode/yolov4-deepsort) ์ฐธ๊ณ ํ  ๊ฒ  
scout-mini์ ๋ํด ๋ ์์ธํ ์๊ณ  ์ถ๋ค๋ฉด [์ฌ๊ธฐ](ttps://github.com/agilexrobotics/scout_mini_ros) ์นจ๊ณ ํ  ๊ฒ  

## 3. ์ต์ข ๊ฒฝ๋ก(์์ฝ)  
catkin_ws(ํด๋ ์ด๋ฆ์ ๋ณ๊ฒฝ ๊ฐ๋ฅ)  
โbuild  
โdevel/setup.bash  
โsrc  
ใโscout_mini_ros  
ใใโscout_bringup  
ใใใโcore  
ใใใโdata/yolov4.weights(or yolov4-tiny.weights)  
ใใใโdeep_sort  
ใใใโlaunch  
ใใใโmodel_data  
ใใใโoutputs  
ใใใโscripts  
ใใใโtools  
ใใใโDefault_dist.py --> ๊น์ด ์ด๊น๊ฐ ์ธก์  (์ด๋ฅผ ํ ๋๋ก ์ฅ์ ๋ฌผ ์์ญ์ ๊น์ด๋ฅผ ์ธก์ ํด ์ฅ์ ๋ฌผ ์ ๋ฌด๋ฅผ ํ๋จํ  ์ ์๋ค.)    
ใใใโcamera.py --> depth camera๋ฅผ ์ด์ฉํ  ์ ์๊ฒ ํ๋ class code  
ใใใโconvert_tflite.py  
ใใใโconvert_trt.py  
ใใใโdrive.py --> ์๋ ฅ ์ด๋ฏธ์ง์ ๋ํ ์ฃผํ ์๊ณ ๋ฆฌ์ฆ(depth๊ฐ๊ณผ RGB๊ฐ์ด ์๋ ฅ๋์ด ์ ์ง/์ ์ง/์ฐํ์ /์ขํ์ /์๋๊ฐ์ ๋ฑ์ ์ ํ๋ค)    
ใใใโkey_move.py --> ์ถ์  & ์ฃผํ ์๊ณ ๋ฆฌ์ฆ์ ๊ฑฐ์ณ ๋์จ ๊ฒฐ๊ณผ๊ฐ(string)์ ๋ฐ๋ผ ์๋์ ๋ฐฉํฅ์ ๋ณ๊ฒฝํด์ฃผ๋ ๋ฉ์๋    
ใใใโobject_track_one_person.py --> ์๋ ฅ ์ด๋ฏธ์ง์ ๋ํ ์ถ์  ์ค์  
ใใใโsave_model.py  
ใใใโscout_motor_light_pub.py --> key_move.py์์ ๋์จ ๊ฒฐ๊ณผ๋ฅผ ROS topic์ผ๋ก ๋ฐํํ๋ ์ฝ๋(๋ชจํฐ ๋ฐ ์กฐ๋ช ์ ์ด)  
ใใใโutils2.py --> ๊น์ด๊ฐ์ ์ด์ฉํด ์ฌ๋๊ณผ์ ๊ฑฐ๋ฆฌ ๋ฐ ์ฅ์ ๋ฌผ ์์ญ ์ธก์     
   
## 4. ์ฌ์ฉ ๋ฐฉ๋ฒ
- `scout_bringup/object_track_one_person.py` ๋ฅผ rosrun ํ๋ฉด ๋๋ค.
1. $ cd catkin_ws/src && source devel/setup/bash  
2. $ roslaunch scout_bringup scout_minimal.launch  
3. ์๋ก์ด ํฐ๋ฏธ๋ ์ด๊ธฐ
4. $ cd catkin_ws/src && source devel/setup/bash  
5. $ rosrun scout_bringup object_track_one_person.py

--> ์์ ์ ์ต์ด1์ธ์ ์ถ์ ํ๋ ์ฝ๋

## 5. ๋ชจ๋ ํ์ผ ์ค๋ช(scout_bringup ํด๋ ๋ด์ ์์)
1. key_move.py --> ์ถ์  & ์ฃผํ ์๊ณ ๋ฆฌ์ฆ์ ๊ฑฐ์ณ ๋์จ ๊ฒฐ๊ณผ๊ฐ(string)์ ๋ฐ๋ผ ์๋์ ๋ฐฉํฅ์ ๋ณ๊ฒฝํด์ฃผ๋ ๋ฉ์๋  
2. scout_motor_light_pub.py --> key_move.py์์ ๋์จ ๊ฒฐ๊ณผ๋ฅผ ROS topic์ผ๋ก ๋ฐํํ๋ ์ฝ๋(๋ชจํฐ ๋ฐ ์กฐ๋ช ์ ์ด)  
3. camera.py --> depth camera๋ฅผ ์ด์ฉํ  ์ ์๊ฒ ํ๋ class code  
4. drive.py --> ์๋ ฅ ์ด๋ฏธ์ง์ ๋ํ ์ฃผํ ์๊ณ ๋ฆฌ์ฆ(depth๊ฐ๊ณผ RGB๊ฐ์ด ์๋ ฅ๋์ด ์ ์ง/์ ์ง/์ฐํ์ /์ขํ์ /์๋๊ฐ์ ๋ฑ์ ์ ํ๋ค)  
5. utils2.py --> ๊น์ด๊ฐ์ ์ด์ฉํด ์ฌ๋๊ณผ์ ๊ฑฐ๋ฆฌ ๋ฐ ์ฅ์ ๋ฌผ ์์ญ ์ธก์   
6. Default_dist.py --> ๊น์ด ์ด๊น๊ฐ ์ธก์  (์ด๋ฅผ ํ ๋๋ก ์ฅ์ ๋ฌผ ์์ญ์ ๊น์ด๋ฅผ ์ธก์ ํด ์ฅ์ ๋ฌผ ์ ๋ฌด๋ฅผ ํ๋จํ  ์ ์๋ค.)  
7. object_track_one_person.py --> ์๋ ฅ ์ด๋ฏธ์ง์ ๋ํ ์ถ์  ์ค์  


**์ฐธ๊ณ **   
์์ฑ๋ code๋ 2๊ฐ์ง ๋ฒ์ ์ด ์์ผ๋ ์ ์ํ  ๊ฒ.  

ํ์ฌ ์ ์ฅ์์ ์์ฑ๋ ์ฝ๋๋ main code2์ด๋ฉฐ, ์ด๊ธฐ์ ํ์ง๋ 1์ธ์ ์ถ์ ํ๋ฉฐ, target lost๊ฐ ๋๋ฉด ์กฐ๋ช์ด blink๋์ด ์ํ๋ฅผ ์๋ ค์ค ์ ์๊ณ  ์ฌ์ธ์์ด ๊ฐ๋ฅํ๋ค.  
main code1๋ ๋ฅํ์ด๋ฅผ ์ฐฉ์ฉํ 1์ธ์ ์ถ์ ํ๋ ์ฝ๋์ด๋ค.   
