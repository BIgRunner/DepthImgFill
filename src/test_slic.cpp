#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


#include "SLIC.hpp"
using namespace std;

int main(int argc, char *argv[])
{
    Mat image = cv::imread(argv[1], 1);
    // Mat labimage = image.clone();
    Mat depth = imread(argv[2], 0);
    Mat labimage;
    cvtColor(image, labimage, CV_BGR2Lab);
    // cout << "2" << endl;
    // cout << labimage.type() << endl;
    // cout<< (int)labimage.ptr<uchar>(0)[0]<<endl;
    // cout << (int)labimage.ptr<uchar>(0)[1] << endl;
    // cout << (int)labimage.ptr<uchar>(0)[2] << endl;
    // cout << "3" << endl;

    Mat depth_fill = depth.clone();
    int w = image.cols;
    int h = image.rows;
    int nr_supeepixels = atoi(argv[3]);

    int step = sqrt((w * h) / (double)nr_supeepixels);

    Slic slic;
    slic.generate_superpixels(labimage, step);
    slic.enforce_connectivity(labimage, depth_fill);

    slic.display_contours(image, CV_RGB(255, 0, 0));
    namedWindow("result");
    imshow("result", image );
    namedWindow("one");
    imshow("one", depth);
    namedWindow("two");
    imshow("two", depth_fill);
    stringstream s;
    s << "../data/" << argv[3] << "_cu.png";
    string filename;
    s >> filename;
    imwrite(filename, depth_fill);
    cvWaitKey(0);
    return 0;

}
