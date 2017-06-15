#include "SLIC.hpp"
using namespace cv;

const int nr_iter = 10;
// Initialize the labels and distance matrices;
void Slic::init_paras(Mat &img)
{
    labels = Mat(img.size(), CV_16UC1, Scalar(65535));
    distance = Mat(img.size(), CV_32FC1, Scalar(FLT_MAX));

    centers = Mat(((img.rows-step/2-1)/step+1)*((img.cols-step/2-1)/step+1),
            6, CV_32FC1, Scalar(0));

    int k = 0;

    for (int r=step/2; r < img.rows ; r += step)
        for (int c=step/2; c < img.cols ; c += step)
        {
            CvPoint nc = find_local_minimum(img, cvPoint(r, c));
            centers.ptr<float>(k)[0] = (float)nc.x;
            centers.ptr<float>(k)[1] = (float)nc.y;
            centers.ptr<float>(k)[2] = (float)img.ptr<uchar>(nc.x)[3*nc.y];
            centers.ptr<float>(k)[3] = (float)img.ptr<uchar>(nc.x)[3*nc.y+1];
            centers.ptr<float>(k)[4] = (float)img.ptr<uchar>(nc.x)[3*nc.y+2];
            centers.ptr<float>(k)[5] = 0.0;
            k++;
        }


    invxywt = 1.0/( 2*step*step);

    maxlab = Mat(k, 1, CV_32FC1, Scalar(5*5));

    new_maxlab = maxlab.clone();
    // cout << centers << endl;
}

double Slic::compute_dist(CvPoint pixel, int ci, Mat &img)
{
    double ds = pow(centers.ptr<float>(ci)[0] - pixel.x, 2) +
                pow(centers.ptr<float>(ci)[1] - pixel.y, 2);
    double dc = pow(centers.ptr<float>(ci)[2]
            - img.ptr<uchar>(pixel.x)[3 * pixel.y],2) +
                pow(centers.ptr<float>(ci)[3]
            - img.ptr<uchar>(pixel.x)[3 * pixel.y + 1],2) +
                pow(centers.ptr<float>(ci)[4]
            - img.ptr<uchar>(pixel.x)[3 * pixel.y + 2],2);

    if (maxlab.ptr<float>(ci)[0] < dc)
        new_maxlab.ptr<float>(ci)[0] = dc;

    return ds*invxywt + dc/maxlab.ptr<float>(ci)[0];
}

bool withInBound(int x,int  y, Mat& img)
{
    return x >= 0 && x < img.rows && y >= 0 && y < img.cols;
}


CvPoint Slic::find_local_minimum(Mat &img, CvPoint center)
{
    double min_grad = FLT_MAX;
    CvPoint loc_min = cvPoint(center.x, center.y);

    for(int r = center.x-1; r < center.x+2; r ++)
        for(int c = center.y-1; c < center.y+2; c ++)
        {
            // CvScalar lab1 = cvGet2D(img, r+1, c);
            // CvScalar lab2 = cvGet2D(img, r, c+1);
            // CvScalar lab3 = cvGet2D(img, r, c);

            if (withInBound(r, c+1, img) && withInBound(r+1,c,img))
            {

                double l1 = img.ptr<uchar>(r+1)[3*c];
                double l2 = img.ptr<uchar>(r)[3*(c+1)];
                double l3 = img.ptr<uchar>(r)[3*c];

                if ((pow(l1-l3,2) + pow(l2-l3,2)) < min_grad)
                {
                    min_grad = (pow(l1-l3,2) + pow(l2-l3,2));
                    loc_min.x = r;
                    loc_min.y = c;
                }
            }
        }

    return loc_min;
}


void Slic::generate_superpixels(Mat &img, int step)
{
    this -> step = step;

    init_paras(img);

    for(int i = 0; i < nr_iter; i++)
    {

        distance = Mat(img.size(), CV_32FC1, FLT_MAX);

        for (int j = 0; j < centers.rows; j++)
        {
            for (int r = centers.ptr<float>(j)[0] - step - 2;
                    r < centers.ptr<float>(j)[0] + step + 2; r++)
                for(int c = centers.ptr<float>(j)[1] - step;
                        c < centers.ptr<float>(j)[1] + step; c++)
                {
                    if (withInBound(r, c, img))
                    {
                        double dist = compute_dist(cvPoint(r, c), j, img);
                        if (dist < distance.ptr<float>(r)[c])
                        {
                            distance.ptr<float>(r)[c] = dist;
                            labels.ptr<ushort>(r)[c] = j;
                        }
                    }
                }
        }

        centers *= 0;
        for (int r = 0; r < img.rows; r++)
            for (int c = 0; c < img.cols; c++)
            {
                int label = labels.ptr<ushort>(r)[c];

                centers.ptr<float>(label)[0] += r;
                centers.ptr<float>(label)[1] += c;
                centers.ptr<float>(label)[2] += img.ptr<uchar>(r)[3*c];
                centers.ptr<float>(label)[3] += img.ptr<uchar>(r)[3*c+1];
                centers.ptr<float>(label)[4] += img.ptr<uchar>(r)[3*c+2];
                centers.ptr<float>(label)[5] ++;

                maxlab = new_maxlab.clone();

            }


        for (int k = 0; k < centers.rows; k++)
        {
            centers.ptr<float>(k)[0] /= centers.ptr<float>(k)[5];
            centers.ptr<float>(k)[1] /= centers.ptr<float>(k)[5];
            centers.ptr<float>(k)[2] /= centers.ptr<float>(k)[5];
            centers.ptr<float>(k)[3] /= centers.ptr<float>(k)[5];
            centers.ptr<float>(k)[4] /= centers.ptr<float>(k)[5];
        }
    }

    // cout << centers << endl;

    namedWindow("x");
    imshow("x", labels);
    cvWaitKey(0);
}

void makeMatFromPoint(vector<Point3d>& withDepth, Mat& A, Mat& b)
{
    A.create(withDepth.size(), 10, CV_32FC1);
    b.create(withDepth.size(), 1, CV_32FC1);
    int k=0;
    for(vector<Point3d>::iterator iter = withDepth.begin();
            iter != withDepth.end(); iter ++)
    {
        A.ptr<float>(k)[0]=1.0;
        A.ptr<float>(k)[1]=iter->x;
        A.ptr<float>(k)[2]=iter->y;
        A.ptr<float>(k)[3]=iter->x * iter->x;
        A.ptr<float>(k)[4]=iter->x * iter->y;
        A.ptr<float>(k)[5]=iter->y * iter->y;

        A.ptr<float>(k)[6] = iter->x *iter->x *iter->x;
        A.ptr<float>(k)[7] = iter->x *iter->x *iter->y;
        A.ptr<float>(k)[8] = iter->x *iter->y *iter->y;
        A.ptr<float>(k)[9] = iter->y *iter->y *iter->y;


        b.ptr<float>(k)[0]=iter->z;

        k++;
    }

}

void fillDepth(vector<Point2i>& withoutDepth, Mat& x, Mat &depth)
{
    for(vector<Point2i>::iterator iter = withoutDepth.begin();
            iter != withoutDepth.end(); iter++)
    {
        depth.ptr<uchar>(iter->x)[iter->y] =
              x.ptr<float>(0)[0]
            + x.ptr<float>(1)[0] * iter -> x
            + x.ptr<float>(2)[0] * iter -> y

            + x.ptr<float>(3)[0] * iter -> x * iter -> x
            + x.ptr<float>(4)[0] * iter -> x * iter -> y
            + x.ptr<float>(5)[0] * iter -> y * iter -> y

            + x.ptr<float>(6)[0] * iter->x * iter->x * iter->x
            + x.ptr<float>(7)[0] * iter->x * iter->x * iter->y
            + x.ptr<float>(8)[0] * iter->x * iter->y * iter->y
            + x.ptr<float>(9)[0] * iter->y * iter->y * iter->y
            + 0.5;
    }

}

void Slic::enforce_connectivity(Mat &img, Mat& depth)
{
    ushort label = 0, adjlabel = 0;
    const int lims = (img.rows * img.cols)/((int)centers.rows);
    const int dx4[4]={-1, 0 , 1, 0};
    const int dy4[4]={0, -1, 0, 1};

    Mat new_labels(labels.size(), CV_16UC1, Scalar(centers.rows+1));

    for (int r = 0; r < img.rows; r ++)
        for (int c = 0; c < img.cols; c ++)
        {
            if (new_labels.ptr<ushort>(r)[c] == centers.rows +1)
            {
                vector<CvPoint> elements;
                elements.push_back(cvPoint(r, c));
                vector<Point3d> withDepth;
                vector<Point2i> withoutDepth;

                new_labels.ptr<ushort>(r)[c] = label;
                for (int n=0; n < 4; n++)
                {
                    int x = r + dx4[n];
                    int y = c + dy4[n];
                    if (withInBound(x, y, img))
                    {
                        if (new_labels.ptr<ushort>(x)[y] != centers.rows + 1)
                            adjlabel = new_labels.ptr<ushort>(x)[y];
                    }
                }

                int count = 1;

                if((int)depth.ptr<uchar>(r)[c] < 10)
                    withoutDepth.push_back(Point2i(r, c));
                else
                {
                    withDepth.push_back(Point3d(r, c,
                                depth.ptr<uchar>(r)[c]));
                }

                for (int m=0; m < count; m++)
                {
                    for ( int n=0; n<4; n++ )
                    {
                        int x = elements[m].x + dx4[n];
                        int y = elements[m].y + dy4[n];

                        if(withInBound(x, y, img))
                            if (new_labels.ptr<ushort>(x)[y] == centers.rows + 1
                                    && labels.ptr<ushort>(x)[y] ==
                                    labels.ptr<ushort>(r)[c])
                            {
                                elements.push_back(cvPoint(x, y));
                                new_labels.ptr<ushort>(x)[y] = label;
                                count ++;


                                if((int)depth.ptr<uchar>(x)[y] < 10)
                                {
                                    withoutDepth.push_back(Point2i(x, y));
                                }
                                else
                                {
                                    withDepth.push_back(Point3d(x, y,
                                                depth.ptr<uchar>(x)[y]));

                                }
                            }
                    }
                }


                if (count < lims >> 2)
                {
                    for (int m = 0; m < count; m++)
                    {
                        new_labels.ptr<ushort>(elements[m].x)[elements[m].y]
                            = adjlabel;
                    }
                    label--;
                    withDepth.clear();
                    withoutDepth.clear();
                }
                else if(withDepth.size() > 40 && withoutDepth.size() != 0)
                {
                    Mat A, b, x;
                    makeMatFromPoint(withDepth, A, b);
                    solve(A, b, x, DECOMP_SVD);
                    // cout << "x = " << x << endl;
                    fillDepth(withoutDepth, x, depth);
                }
                label++;
            }

        }
    labels = new_labels.clone();
}

void Slic::display_center_grid(Mat &img, CvScalar color)
{
    for (int i = 0; i < centers.rows; i++)
    {
        circle(img, cvPoint(centers.ptr<uchar>(i)[0],
                    centers.ptr<uchar>(i)[1]), 2, color, 2);
    }
}

void Slic::display_contours(Mat &img, CvScalar color)
{
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    vector<CvPoint> contours;
    Mat istaken = Mat(img.size(), CV_8UC1, Scalar(0));

    for (int r=0; r<img.rows; r++)
        for (int c=0; c<img.cols; c++)
        {
            int nr_p = 0;

            for(int k = 0; k < 8; k++)
            {
                int x = r + dx8[k];
                int y = c + dy8[k];

                if ( withInBound(x, y, img) )
                {
                    if (istaken.ptr<uchar>(x)[y] == 0 &&
                            labels.ptr<ushort>(x)[y]!=labels.ptr<ushort>(r)[c])
                        nr_p ++;
                }
            }

            if(nr_p >= 2)
            {
                contours.push_back(cvPoint(r,c));
                istaken.ptr<uchar>(r)[c] = 255;
            }
        }

    for (vector<CvPoint>::iterator iter = contours.begin();
            iter != contours.end(); iter ++)
    {
        // cvGet2D(img, iter->x, iter->y, color);
        img.ptr<uchar>(iter->x)[3 * iter->y] = color.val[0];
        img.ptr<uchar>(iter->x)[3 * iter->y + 1] = color.val[1];
        img.ptr<uchar>(iter->x)[3 * iter->y + 2] = color.val[2];

    }
}


void Slic::color_with_cluster_means(Mat &img)
{
    for (int r=0; r < img.rows; r++)
        for (int c=0; c < img.cols; c++)
        {
            int label = labels.ptr<ushort>(r)[c];
            img.ptr<uchar>(r)[3*c] = centers.ptr<uchar>(label)[2];
            img.ptr<uchar>(r)[3*c +1] = centers.ptr<uchar>(label)[3];
            img.ptr<uchar>(r)[3*c +2 ] = centers.ptr<uchar>(label)[4];
        }
}
