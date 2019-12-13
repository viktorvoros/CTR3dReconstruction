#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def webcam_pub():
    pub = rospy.Publisher('camera2', Image, queue_size=1)
    rospy.init_node('webcam2_pub', anonymous=True)
    rate = rospy.Rate(20) # 20hz

    cam = cv2.VideoCapture(1)
    bridge = CvBridge()

    if not cam.isOpened():
         sys.stdout.write("Webcam2 is not available")
         return -1

    while not rospy.is_shutdown():
        ret, frame = cam.read()


    msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

    if ret:
        rospy.loginfo("Capturing image 2 failed.")


    pub.publish(msg)
    rate.sleep()


if __name__ == '__main__':
    try:
        webcam_pub()
    except rospy.ROSInterruptException:
      pass
