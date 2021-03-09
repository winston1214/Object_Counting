## People Counting

### Environment

- Ubuntu 20.04
- GPU : NVIIDA RTX 3090 x2

### Reference
<a href='https://github.com/ultralytics/yolov5'>Yolov5</a>, <a href='https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch'>Yolov5 DeepSort</a>

### Description
**Algorithms for calculating the number of people and cars in CCTV.**

**You can extract statistics on how many people or cars there were by time zone.**

### How to do?

```
$ git clone https://github.com/winston1214/Car_Counting && cd Car_Counting
```

```
$ git clone https://github.com/ultralytics/yolov5
```

```
& pip install -r requirements.txt
```
⛔ If you occur error, If an error occurs, enter the requirements.txt file and comment on the packages that may have a version conflict.

⛔ If the equipment is Nvidia mini PC(ex. Jetson Nano, Xavier), the pytorch is not installed as a pip.

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
  - daily count : ```if time_sum>=60*60*24```

2. **If you want to reposition the line, fix <a href='https://github.com/winston1214/Car_Counting/blob/223f7a076a7f1b932fff582f5f809b15ed95ca75/track.py#L168'>this part</a>**
3. **If you want to reposition the text, fix <a href='https://github.com/winston1214/Car_Counting/blob/223f7a076a7f1b932fff582f5f809b15ed95ca75/track.py#L281'>this part</a>**
