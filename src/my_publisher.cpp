#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

std::string full_int_str(int index, char a, int length)
{
    std::string s;
    std::stringstream tmp;
    tmp << index;
    tmp >> s;
    while(s.length() < length)
    {
        s = a + s;
    }
    return s;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: rosrun cv_bridge_use publisher dir_of_image startindex endindex" << std::endl;
        return -1;
    }

    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_1 = it.advertise("camera/color", 1);
    image_transport::Publisher pub_2 = it.advertise("camera/depth", 1);
    cv::Mat color, depth;
    std::string color_filename, depth_filename;
    sensor_msgs::ImagePtr msg1, msg2;

    ros::Rate loop_rate(5);
    std::string root_dir = argv[1];
    int startindex = atoi(argv[2]);
    int endindex = atoi(argv[3]);

    int index = startindex;

    while (nh.ok())
    {
        color_filename = root_dir + "color/" + full_int_str(index, '0', 5) + "-color.png";
        depth_filename = root_dir + "depth/" + full_int_str(index, '0', 5) + "-depth.png";
        color = cv::imread(color_filename, 1);
        depth = cv::imread(depth_filename, 0);
        msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color).toImageMsg();
        msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", depth).toImageMsg();


        pub_1.publish(msg1);
        pub_2.publish(msg2);
        ros::spinOnce();
        loop_rate.sleep();
        index ++;
    }
}
