# -*- encoding: UTF-8 -*-

''' Whole Body Motion: Left or Right Arm position control '''
''' This example is only compatible with NAO '''

import argparse
import motion
import time
from naoqi import ALProxy

def main(robotIP, PORT):
    ''' Example of a whole body Left or Right Arm position control
        Warning: Needs a PoseInit before executing
                 Whole body balancer must be inactivated at the end of the script
    '''

    motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)

    # Wake up robot
    motionProxy.wakeUp()

    # Go to rest position
    #motionProxy.rest()
    names = list()
    times = list()
    keys = list()

    names.append("LShoulderPitch")
    times.append([0.72, 1.48, 2.16, 2.72, 3.4, 4.52])
    keys.append([1.11824, 0.928028, 0.9403, 0.862065, 0.897349, 0.842125])

    names.append("RShoulderPitch")
    times.append([0.72, 1.48, 2.16, 2.72, 3.4, 4.52])
    keys.append([1.11824, 0.928028, 0.9403, 0.862065, 0.897349, 0.842125])

    try:
      # uncomment the following line and modify the IP if you use this script outside Choregraphe.
      # motion = ALProxy("ALMotion", IP, 9559)
      #motion = ALProxy("ALMotion")
      motionProxy.angleInterpolation(names, keys, times, True)
    except BaseException, err:
      print err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")

    args = parser.parse_args()
    main(args.ip, args.port)