#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

Mat bina(Mat src, int k)
{
    Mat src1 = src.clone();
    int rownum = src1.rows;
    int colnum = src1.cols;
    for (int i = 0; i < rownum; i++)
    {
        for (int j = 0; j < colnum; j++)
        {
            if (src1.at<uchar>(i, j) < k)
            {
                src1.at<uchar>(i, j) = 0;
            }
            else
            {
                src1.at<uchar>(i, j) = 255;
            }
        }
    }
    return src1;
}
Mat logcg(Mat src, float c)
{
    Mat src1 = src.clone();
    int rownum = src1.rows;
    int colnum = src1.cols;
    for (int i = 0; i < rownum; i++)
    {
        for (int j = 0; j < colnum; j++)
        {
            src1.at<uchar>(i, j) = c * log(1 + src1.at<uchar>(i, j)) * (255 / log(256));
        }
    }
    normalize(src1, src1, 0, 255, CV_MINMAX);
    return src1;
}

Mat gamacg(Mat src, float c)
{
    Mat src1 = src.clone();
    int rownum = src1.rows;
    int colnum = src1.cols;
    for (int i = 0; i < rownum; i++)
    {
        for (int j = 0; j < colnum; j++)
        {
            src1.at<uchar>(i, j) = std::pow(src1.at<uchar>(i, j), c) * (255 / pow(255, c)); //后面(255/pow（255，c）是为了进行归一化处理)
        }
    }
    normalize(src1, src1, 0, 255, CV_MINMAX);
    return src1;
}

Mat colcg(Mat src)
{
    Mat src1 = src.clone();
    int rownum = src1.rows;
    int colnum = src1.cols;
    for (int i = 0; i < rownum; i++)
    {
        for (int j = 0; j < colnum; j++)
        {
            src1.at<Vec3b>(i, j)[0] = 255 - src1.at<Vec3b>(i, j)[0]; //对每个像素进行反向变换
            src1.at<Vec3b>(i, j)[1] = 255 - src1.at<Vec3b>(i, j)[1];
            src1.at<Vec3b>(i, j)[2] = 255 - src1.at<Vec3b>(i, j)[2];
        }
    }
    return src1;
}

int main1()
{
    int threshold1;
    float threshold2, threshold3;
    Mat colimg = imread("demo.jpg", 1);
    Mat img = imread("demo.jpg", 0);

    std::cout << "二项化阈值参数" << std::endl;
    std::cin >> threshold1;
    std::cout << "对数变换的阈值参数" << std::endl;
    std::cin >> threshold2;
    std::cout << "伽马变换的伽马参数" << std::endl;
    std::cin >> threshold3;
    namedWindow("原图");
    namedWindow("二项化");
    namedWindow("对数");
    namedWindow("伽马");
    namedWindow("补色");
    Mat img2 = bina(img, threshold1);
    Mat img3 = logcg(img, threshold2);
    Mat img4 = gamacg(img, threshold3);
    Mat img5 = colcg(img);
    imshow("二项化", img2);
    imshow("对数", img3);
    imshow("伽马", img4);
    imshow("原图", colimg);
    imshow("补色", img5);
    waitKey();
    return 1;
}