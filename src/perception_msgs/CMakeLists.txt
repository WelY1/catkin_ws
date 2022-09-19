cmake_minimum_required(VERSION 3.0.2)
project(perception_msgs)

find_package(catkin REQUIRED 
  geometry_msgs
  uuid_msgs
  message_generation
  jsk_recognition_msgs
  sensor_msgs
  std_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
    DynamicObject.msg
    DynamicObjectArray.msg
    Shape.msg
    Semantic.msg
    State.msg
    Feature.msg
    PredictedPath.msg
    DynamicObjectWithFeature.msg
    DynamicObjectWithFeatureArray.msg
    PredictedPath.msg
)



generate_messages(
  DEPENDENCIES
    geometry_msgs
    jsk_recognition_msgs
    uuid_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    jsk_recognition_msgs
    uuid_msgs
    message_runtime
    sensor_msgs
    std_msgs
)
