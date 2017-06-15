#ifndef SLIC_OpenCV_SLIC_HPP
#define SLIC_OpenCV_SLIC_HPP

/* SLIC.HPP
 * written by BigRunner
 * This hpp file contains class SLIC, which is used for image segmantation.
 * more details in
 */


#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Slic
{
    private:
        // labels for each pixel;
        Mat labels;
        // Distance for each pixel;
        Mat distance;

        // Cluster centers contains XY and LAB value;
        Mat centers;

        // the step of each cluster, the cluster parameter m between 1 to 60;
        int  step;
        double invxywt;

        Mat maxlab, new_maxlab;

        // compute distance between one pixel to cluster center.
        double compute_dist(CvPoint pixel, int ci, Mat &img);

        // Find the pixel with the lowest gradient in a 3x3 surrounding.
        CvPoint find_local_minimum(Mat &img, CvPoint center);

        // initialize parameters with image.
        void init_paras(Mat &img);

        // The iterations in cluster algorithm.

    public:
        // ocnstructors and deconstructor.
        // Slic();:
        // ~Slic();

        // Generate superpixels in an image
        void generate_superpixels(Mat &image, int step);

        // Enforce connectivity for an image
        void enforce_connectivity(Mat &img, Mat& depth);

        // Display function
        void display_center_grid(Mat &img, CvScalar color);
        void display_contours(Mat &img, CvScalar color);
        void color_with_cluster_means(Mat &image);
};

#endif
