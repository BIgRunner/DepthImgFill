#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

void colorCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv::imshow("color", cv_bridge::toCvShare(msg, "bgr8") -> image);
        cv::waitKey(30);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' t 'bgr8'.", msg->encoding.c_str());
    }
}

void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv::imshow("depth", cv_bridge::toCvShare(msg, "mono8") -> image);
        cv::waitKey(30);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' t 'mono8'.", msg->encoding.c_str());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("color");
    cv::namedWindow("depth");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub1 = it.subscribe("camera/color", 1, colorCallback);
    image_transport::Subscriber sub2 = it.subscribe("camera/depth", 1, depthCallback);
    ros::spin();
}
