import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="./yolov4-tiny_last.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./voc.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)    # 608
    height = darknet.network_height(network)   # 608
       
    darknet_image = darknet.make_image(width, height, 3)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)   
    # detections: list of ('Car', '35.68', (72.75263977050781, 379.9039306640625, 134.5846710205078, 90.21629333496094))x,y,w,h
    
    darknet.free_image(darknet_image)
    image_box = darknet.draw_boxes(detections, image_resized, class_colors)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
    return image, detections


def main():
    args = parser()

    # random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    image = cv2.imread('/home/zxc/yutong/yolov5/data/images/luce/1.jpg')

   
    prev_time = time.time()
    
    print(image.shape,'-->')
    
    image, detections = image_detection(
        image, network, class_names, class_colors, args.thresh
        )
    print(image.shape)
    
    # darknet.print_detections(detections, False)
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))
    cv2.imshow('Inference', image)
    cv2.waitKey(5000)



if __name__ == "__main__":
    main()
