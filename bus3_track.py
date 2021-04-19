import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import datetime

import json
import requests

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None,labels=None, offset=(0, 0)):
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        names = int(labels[i]) if labels is not None else 0
        dic = {0:'person',2:'car'}
        obj_names = dic[names]
        # color = compute_color_for_labels(id)
        if obj_names == 'person':
            color = (0,0,255) # red
        if obj_names == 'car':
            color = (0,255,255) # yellow
        label = '{}{:d}'.format(obj_names, id)
        
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    now = datetime.datetime.now().strftime("%Y/%m/%d/%H:%M:%S") # current time

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = False
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    url = 'https://api.blackstonebelleforet.com/count/peoplecount'
    uid = 'bus1'
    os.system('shutdown -r 06:00')
    memory = {}
    people_counter = 0
    car_counter = 0
    in_people = 0
    out_people = 0
    time_sum = 0
    now_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            img_center_x = int(im0.shape[1]//2)
            # line = [(0,img_center_y),(im0.shape[1],img_center_y)]
            line = [(int(img_center_x + 150),0),(img_center_x+50,int(im0.shape[0]))]
            line2 = [(int(img_center_x + 200),0),(img_center_x+170,int(im0.shape[0]))]
            cv2.line(im0,line[0],line[1],(0,0,255),5)
            cv2.line(im0,line2[0],line2[1],(0,255,0),5)
          
            
            if det is not None and len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                crop_xyxy = det[:,:4]
                det = det[crop_xyxy[:,0]<img_center_x + 200] # line 오른쪽 지우기
                if len(det) == 0:
                    pass
                else:

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                    bbox_xywh = []
                    confs = []
                    bbox_xyxy = []


                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)

                        
                        obj = [x_c, y_c, bbox_w, bbox_h,int(cls)]
                    
                        #cv2.circle(im0,(int(x_c),int(y_c)),color=(0,255,255),radius=12,thickness = 10)
                        bbox_xywh.append(obj)
                        # bbox_xyxy.append(rec)
                        confs.append([conf.item()])
                        


                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0) # deepsort
                    index_id = []
                    previous = memory.copy()
                    memory = {}
                    boxes = []
                    names_ls = []



                    # draw boxes for visualization
                    if len(outputs) > 0:
                        
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        labels = outputs[:,-1]
                        dic = {0:'person',2:'car'}
                        for i in labels:
                            names_ls.append(dic[i])
                        
                        # print('output len',len(outputs))
                        for output in outputs:
                            boxes.append([output[0],output[1],output[2],output[3]])
                            index_id.append('{}-{}'.format(names_ls[-1],output[-2]))

                            memory[index_id[-1]] = boxes[-1]

                        if time_sum>=60:
                            param={'In_people':in_people,'Out_people':out_people,'uid':uid,'time':now_time+'~'+datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
                            response = requests.post(url,data=param)
                            response_text = response.text
                            with open('counting.txt','a') as f:
                                f.write('{}~{} IN : {}, Out : {} Response: {}\n'.format(now_time,datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),in_people,out_people,response_text))

                            people_counter,car_counter,in_people,out_people = 0,0,0,0
                            time_sum = 0
                            now_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                        i = int(0)
                        for box in boxes:
                            # extract the bounding box coordinates
                            (x, y) = (int(box[0]), int(box[1]))
                            (w, h) = (int(box[2]), int(box[3]))


                            if index_id[i] in previous:
                                previous_box = previous[index_id[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                
                                cv2.line(im0, p0, p1, (0,255,0), 3) # current frame obj center point - before frame obj center point
                            
                                
                                if intersect(p0, p1, line[0], line[1]) and index_id[i].split('-')[0] == 'person':
                                    people_counter += 1
                                    if p0[0] > line[1][0]:
                                        in_people +=1
                                    else:
                                        out_people +=1
                                if intersect(p0, p1, line[0], line[1]) and index_id[i].split('-')[0] == 'car':
                                    car_counter +=1
                                
                                
    
                            i += 1

                        draw_boxes(im0,bbox_xyxy,identities,labels)
                            
                        

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format       

            else:
                deepsort.increment_ages()
            cv2.putText(im0, 'In : {}, Out : {}'.format(in_people,out_people),(130,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)
            cv2.putText(im0, 'Person : {}'.format(people_counter), (130,100),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),3)
            # Print time (inference + NMS)
            if time_sum>=60:
                param={'In_people':in_people,'Out_people':out_people,'uid':uid,'time':now_time+'~'+datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
                response = requests.post(url,data=param)
                response_text = response.text
                with open('counting.txt','a') as f:
                    f.write('{}~{} IN : {}, Out : {}, Response: {}\n'.format(now_time,datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),in_people,out_people,response_text))

                people_counter,car_counter,in_people,out_people = 0,0,0,0
                time_sum = 0
                now_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            time_sum += t2-t1
            


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    im0= cv2.resize(im0,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(save_path, im0)
                else:
                    
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    param={'In_people':in_people,'Out_people':out_people,'uid':uid,'time':now_time+'~'+datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
    response = requests.post(url,data=param)
    response_text = response.text
    with open('counting.txt','a') as f:
        f.write('{}~{} IN : {}, Out : {}, Response: {}\n'.format(now_time,datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),in_people,out_people,response_text))
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    with torch.no_grad():
        detect(args)
