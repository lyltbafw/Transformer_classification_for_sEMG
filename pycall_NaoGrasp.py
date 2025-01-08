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

    names.append("LHand")
    times.append([1.4, 2.2, 3.32, 4.44])
    keys.append([0.853478, 0.654933, 0.425116, 0.240025])

    names.append("RHand")
    times.append([1.4, 2.2, 3.32, 4.44])
    keys.append([0.853478, 0.654933, 0.425116, 0.240025])

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