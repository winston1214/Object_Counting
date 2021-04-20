## People Counting

### Environment

- Ubuntu 20.04
- GPU : NVIIDA RTX 3090 x2

### Reference
<a href='https://github.com/ultralytics/yolov5'>Yolov5</a>, <a href='https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch'>Yolov5 DeepSort</a>

### Description
**Algorithms for calculating the number of people and cars in CCTV.**

**You can extract statistics on how many people or cars there were by time zone.(Now set to every 1 minute)**

### Preparation in advance(After pip install -r requirements.txt)

- Download yolov5 weight file : You can download the weight file <a href='https://github.com/ultralytics/yolov5'>here</a>.
  ```
  Sample Command
  
  $ cd yolov5
  $ python3 detect.py --weights yolov5s.pt
                                yolov5m.pt
                                yolov5l.pt
                                yolov5x.pt
  ```
- Download deepsort file : You can download the weight file <a href='https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6'>here</a>.
                          And place ckpt.t7 file under pytorch_deep_sort/deep_sort/deep/checkpoint/


- If you use **Jetson nano**, you should change **dataset.py to yolov5/utils/dataset.py**
### How to do?

```
$ git clone https://github.com/winston1214/Car_Counting && cd Car_Counting
```
```
$ rm yolov5
```
```
$ git clone https://github.com/ultralytics/yolov5
```
```
$ pip3 install -r requirements.txt
```
⛔ If you occur error enter the requirements.txt file and comment on the packages that may have a version conflict.

⛔ If the equipment is Nvidia mini PC(ex. Jetson Nano, Xavier), the pytorch is not installed as a pip.

⛔ In jetson nano mini pc, you can enter some commands
```
$ sudo gedit ~/.bashrc
```
```
# add last line
export OPENBLAS_CORETYPE=ARMV8
```
```
$ source ~/.bashrc
```


**Excute Command**

```
$ python3 track.py --source VIDEO_PATH --weights yolov5/yolov5x.pt --classes 2
```
- classes 0 : Person, classes 2 : Car
- If you want to specify several classes, specify the option Class 0 2(person,car)
  - **But, it's unstable**

### Tunning Point

1. **If you want to see the daily and hourly counts, fix <a href='https://github.com/winston1214/Car_Counting/blob/223f7a076a7f1b932fff582f5f809b15ed95ca75/track.py#L233'>this part</a>.**

- Example
  - hour count : ```if time_sum>=60*60:```
  - daily count : ```if time_sum>=60*60*24:```

2. **If you want to reposition the line, fix <a href='https://github.com/winston1214/Car_Counting/blob/223f7a076a7f1b932fff582f5f809b15ed95ca75/track.py#L168'>this part</a>**
3. **If you want to reposition the text, fix <a href='https://github.com/winston1214/Car_Counting/blob/223f7a076a7f1b932fff582f5f809b15ed95ca75/track.py#L281'>this part</a>**

4. **Bus Revise**
  - **Adjust Undetectable area**  <a href='https://github.com/winston1214/Car_Counting/blob/5785f303d622b2bceb5a3a962d5aac1bde5e5605/bus1_track.py#L181'>this part</a>
  - **Visualize Line** <a href='https://github.com/winston1214/Car_Counting/blob/5785f303d622b2bceb5a3a962d5aac1bde5e5605/bus1_track.py#L170'>this part</a>              
           




### Sample Output

<center><a href="https://youtu.be/8sUBoE8mZG4" target="_blank"><img src='https://i9.ytimg.com/vi/8sUBoE8mZG4/mq1.jpg?sqp=CMTeqIIG&rs=AOn4CLBd7W5OpDxRjEd-eAfeeQ1qOj9Ahw'
alt="Car Counting" /></a></center>
