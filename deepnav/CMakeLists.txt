cmake_minimum_required(VERSION 2.8.3)
project(deepnav)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  gmapping 
  map_server 
  amcl 
  move_base
  roscpp
  rospy
  std_msgs
)

catkin_package()

