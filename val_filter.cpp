#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.1415926
#define SIZE 11

using namespace std;
using namespace cv;

String path = "demo.jpg";

void Highfilter(int x, int y)
{
    Mat src;
    Mat mean_filter(x, y, CV_8UC1, Scalar(1)); //创建均值卷积模版
    src = imread(path, 0);
    Mat img = src.clone();

    int rownum = img.rows;
    int colnum = img.cols;

    int temp;
    int filter_count = 0;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            filter_count += mean_filter.at<uchar>(i, j);
        }
    }

    for (int i = 0; i < rownum - x; i++)
    {
        for (int j = 0; j < colnum - y; j++)
        {
            temp = 0;
            for (int k = 0; k < x; k++)
            {
                for (int m = 0; m < y; m++)
                {
                    temp += src.at<uchar>(i + k, j + m) * mean_filter.at<uchar>(k, m);
                }
            }
            temp /= filter_count;
            img.at<uchar>(i + (x - 1) / 2, j + (y - 1) / 2) = temp;
        }
    }
    namedWindow("模糊图", WINDOW_AUTOSIZE);
    imshow("模糊图", img);
    waitKey();
    Mat high = src - img;
    high = src + high;

    namedWindow("高提升滤波", WINDOW_AUTOSIZE);
    imshow("高提升滤波", img);
    waitKey();
}

void Sobelfilter_Color()
{
    Mat src = imread(path, 1);
    Mat image = src.clone();
    namedWindow("ԭͼ", WINDOW_AUTOSIZE);
    imshow("ԭͼ", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int k = 0; k < 3; k++)
    {
        for (int i = 1; i < row_num - 1; i++)
        {

            for (int j = 1; j < col_num - 1; j++)
            {
                image.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(abs((src.at<Vec3b>(i + 1, j - 1)[k] + 2 * src.at<Vec3b>(i + 1, j)[k] + src.at<Vec3b>(i + 1, j + 1)[k]) - (src.at<Vec3b>(i - 1, j - 1)[k] + 2 * src.at<Vec3b>(i - 1, j)[k] + src.at<Vec3b>(i - 1, j + 1)[k])) + abs((src.at<Vec3b>(i - 1, j + 1)[k] + 2 * src.at<Vec3b>(i, j + 1)[k] + src.at<Vec3b>(i + 1, j + 1)[k]) - (src.at<Vec3b>(i - 1, j - 1)[k] + 2 * src.at<Vec3b>(i, j - 1)[k] + src.at<Vec3b>(i + 1, j - 1)[k]))); //f(x)���ݶ�Ϊf(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
            }
        }
    }
    image = image + src;
    namedWindow("索贝尔变换", WINDOW_AUTOSIZE);
    imshow("索贝尔变换", image);
    waitKey();
}

void Sobelfilter_Gray()
{
    Mat src = imread(path, 0);
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int i = 1; i < row_num - 1; i++)
    {

        for (int j = 1; j < col_num - 1; j++)
        {
            image.at<uchar>(i, j) = saturate_cast<uchar>(abs((src.at<uchar>(i + 1, j - 1) + 2 * src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1)) - (src.at<uchar>(i - 1, j - 1) + 2 * src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j + 1))) + abs((src.at<uchar>(i - 1, j + 1) + 2 * src.at<uchar>(i, j + 1) + src.at<uchar>(i + 1, j + 1)) - (src.at<uchar>(i - 1, j - 1) + 2 * src.at<uchar>(i, j - 1) + src.at<uchar>(i + 1, j - 1)))); //f(x)���ݶ�Ϊf(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
        }
    }
    image = image + src;
    namedWindow("索贝尔变换", WINDOW_AUTOSIZE);
    imshow("索贝尔变换", image);
    waitKey();
}

void Robertfilter_Color()
{
    Mat src = imread(path, 1);
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int k = 0; k < 3; k++)
    {
        for (int i = 1; i < row_num - 1; i++)
        {

            for (int j = 1; j < col_num - 1; j++)
            {
                image.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(src.at<Vec3b>(i + 1, j + 1)[k] - src.at<Vec3b>(i, j)[k]);
            }
        }
    }
    image = image + src;
    namedWindow("罗伯特变换", WINDOW_AUTOSIZE);
    imshow("罗伯特变换", image);
    waitKey();
}

void Robertfilter_Gray()
{
    Mat src = imread(path, 0);
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int i = 1; i < row_num - 1; i++)
    {

        for (int j = 1; j < col_num - 1; j++)
        {
            image.at<uchar>(i, j) = saturate_cast<uchar>(src.at<uchar>(i + 1, j + 1) - src.at<uchar>(i, j)); //f(x)���ݶ�Ϊf(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
        }
    }
    image = image + src;
    namedWindow("罗伯特变换", WINDOW_AUTOSIZE);
    imshow("罗伯特变换", image);
    waitKey();
}

void Laplacianfilter_Color()
{
    Mat src = imread(path, 1);
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int k = 0; k < 3; k++)
    {
        for (int i = 1; i < row_num - 1; i++)
        {

            for (int j = 1; j < col_num - 1; j++)
            {
                image.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(src.at<Vec3b>(i - 1, j)[k] + src.at<Vec3b>(i + 1, j)[k] + +src.at<Vec3b>(i, j + 1)[k] + src.at<Vec3b>(i, j - 1)[k] - 4 * src.at<Vec3b>(i, j)[k]); //f(x)���ݶ�Ϊf(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
            }
        }
    }
    image = image + src;
    namedWindow("拉普拉斯变换", WINDOW_AUTOSIZE);
    imshow("拉普拉斯变换", image);
    waitKey();
}

void Laplacianfilter_Gray()
{
    Mat src = imread(path, 0);
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    int row_num = src.rows;
    int col_num = src.cols;

    for (int i = 1; i < row_num - 1; i++)
    {
        uchar *pre = src.ptr<uchar>(i - 1);
        uchar *now = src.ptr<uchar>(i);
        uchar *back = src.ptr<uchar>(i + 1);
        for (int j = 1; j < col_num; j++)
        {
            image.at<uchar>(i, j) = saturate_cast<uchar>(now[j + 1] + now[j - 1] + pre[j] + back[j] - 4 * now[j]); //f(x)���ݶ�Ϊf(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
        }
    }
    //normalize(image, image, 0, 255 );
    image = image + src;
    namedWindow("拉普拉斯变换", WINDOW_AUTOSIZE);
    imshow("拉普拉斯变换", image);
    waitKey();
}
void Gaussianfilter_Color(int x, int y)
{
    Mat src;
    src = imread(path, 1);
    copyMakeBorder(src, src, y / 2, y / 2, x / 2, x / 2, BORDER_CONSTANT, Scalar(0, 0, 0));
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    Mat mean_filter(x, y, CV_8UC1, Scalar(0)); //������˹����ģ��
    double gaussian_fuc[SIZE][SIZE];           //�����޸Ĵ�С�����ڴ����˹������ֵ
    double sigma = 1;                          //�����޸�ֵ
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            gaussian_fuc[i][j] = (1.0 / (2 * PI * sigma)) * exp(-((i - SIZE / 2) * (i - SIZE / 2) + (j - SIZE / 2) * (j - SIZE / 2)) / (2.0 * sigma * sigma)); //�洢��˹������ֵ
        }
    }

    int start_x = (SIZE - x) / 2; //�Ǹ�˹��������ʼλ�ã�Ҳ�ǹ��Ϊ1��λ��
    int start_y = (SIZE - y) / 2;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            mean_filter.at<uchar>(i, j) = uchar(gaussian_fuc[i + start_x][j + start_y] / gaussian_fuc[start_x][start_y]); //���ɸ�˹ģ��
        }
    }

    int rownum = image.rows;
    int colnum = image.cols;

    int temp[3];
    int filter_count = 0;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            filter_count += mean_filter.at<uchar>(i, j); //����ģ��ĺ�
        }
    }
    for (int i = 0; i < rownum - x; i++)
    {
        for (int j = 0; j < colnum - y; j++)
        {
            temp[0] = 0;
            temp[1] = 0;
            temp[2] = 0;
            for (int k = 0; k < x; k++) //����������ֵ
            {
                for (int m = 0; m < y; m++)
                {
                    temp[0] = temp[0] + ((src.at<Vec3b>(i + k, j + m)[0]) * mean_filter.at<uchar>(k, m)); //���о�������
                    temp[1] = temp[1] + ((src.at<Vec3b>(i + k, j + m)[1]) * mean_filter.at<uchar>(k, m));
                    temp[2] = temp[2] + ((src.at<Vec3b>(i + k, j + m)[2]) * mean_filter.at<uchar>(k, m));
                }
            }
            temp[0] = temp[0] / filter_count;
            temp[1] = temp[1] / filter_count;
            temp[2] = temp[2] / filter_count;
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[0] = saturate_cast<uchar>(temp[0]);
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[1] = saturate_cast<uchar>(temp[1]);
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[2] = saturate_cast<uchar>(temp[2]);
        }
    }
    namedWindow("高斯变换", WINDOW_AUTOSIZE);
    imshow("高斯变换", image);
    waitKey();
}

void Gaussianfilter_Gray(int x, int y)
{
    Mat src;
    src = imread(path, 0);
    copyMakeBorder(src, src, y / 2, y / 2, x / 2, x / 2, BORDER_CONSTANT, Scalar(0));
    Mat mean_filter(x, y, CV_8UC1, Scalar(0)); //������˹����ģ��
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);

    double gaussian_fuc[SIZE][SIZE]; //�����޸Ĵ�С�����ڴ����˹������ֵ
    double sigma = 1;                //�����޸�ֵ
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            gaussian_fuc[i][j] = (1.0 / (2 * PI * sigma)) * exp(-((i - SIZE / 2) * (i - SIZE / 2) + (j - SIZE / 2) * (j - SIZE / 2)) / (2.0 * sigma * sigma)); //�洢��˹������ֵ
        }
    }

    int start_x = (SIZE - x) / 2; //�Ǹ�˹��������ʼλ�ã�Ҳ�ǹ��Ϊ1��λ��
    int start_y = (SIZE - y) / 2;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            mean_filter.at<uchar>(i, j) = uchar(gaussian_fuc[i + start_x][j + start_y] / gaussian_fuc[start_x][start_y]); //���ɸ�˹ģ��
        }
    }

    int rownum = image.rows;
    int colnum = image.cols;

    int temp;
    int filter_count = 0;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            filter_count += mean_filter.at<uchar>(i, j);
        }
    }
    for (int i = 0; i < rownum - x; i++)
    {
        for (int j = 0; j < colnum - y; j++)
        {
            temp = 0;
            for (int k = 0; k < x; k++) //����������ֵ
            {
                for (int m = 0; m < y; m++)
                {
                    temp = temp + (src.at<uchar>(i + k, j + m) * mean_filter.at<uchar>(k, m)); //���о�������
                }
            }
            temp = temp / filter_count;
            image.at<uchar>(i + (x - 1) / 2, j + (y - 1) / 2) = saturate_cast<uchar>(temp);
        }
    }
    namedWindow("高斯变换", WINDOW_AUTOSIZE);
    imshow("高斯变换", image);
    waitKey();
}

void Meanfilter_Color(int x, int y)
{
    Mat src;
    Mat mean_filter(x, y, CV_8UC1, Scalar(1)); //������ֵ����ģ��
    src = imread(path, 1);
    copyMakeBorder(src, src, y / 2, y / 2, x / 2, x / 2, BORDER_CONSTANT, Scalar(0, 0, 0));
    Mat image = src.clone();
    namedWindow("ԭͼ", WINDOW_AUTOSIZE);
    imshow("ԭͼ", src);

    int rownum = image.rows;
    int colnum = image.cols;

    int temp[3];
    int filter_count = 0;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            filter_count += mean_filter.at<uchar>(i, j); //����ģ��ĺ�
        }
    }
    for (int i = 0; i < rownum - x; i++)
    {
        for (int j = 0; j < colnum - y; j++)
        {
            temp[0] = 0;
            temp[1] = 0;
            temp[2] = 0;
            for (int k = 0; k < x; k++) //����������ֵ
            {
                for (int m = 0; m < y; m++)
                {
                    temp[0] = temp[0] + ((src.at<Vec3b>(i + k, j + m)[0]) * mean_filter.at<uchar>(k, m)); //���о�������
                    temp[1] = temp[1] + ((src.at<Vec3b>(i + k, j + m)[1]) * mean_filter.at<uchar>(k, m));
                    temp[2] = temp[2] + ((src.at<Vec3b>(i + k, j + m)[2]) * mean_filter.at<uchar>(k, m));
                }
            }
            temp[0] = temp[0] / filter_count;
            temp[1] = temp[1] / filter_count;
            temp[2] = temp[2] / filter_count;
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[0] = temp[0];
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[1] = temp[1];
            image.at<Vec3b>(i + (x - 1) / 2, j + (y - 1) / 2)[2] = temp[2];
        }
    }
    namedWindow("均值变换", WINDOW_AUTOSIZE);
    imshow("均值变换", image);
    waitKey();
}

void Meanfilter_Grey(int x, int y)
{
    Mat src;
    Mat mean_filter(x, y, CV_8UC1, Scalar(1)); //������ֵ����ģ��
    src = imread(path, 0);
    copyMakeBorder(src, src, y / 2, y / 2, x / 2, x / 2, BORDER_CONSTANT, Scalar(0));
    Mat image = src.clone();
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", src);
    cout << mean_filter << endl;

    int rownum = image.rows;
    int colnum = image.cols;

    int temp;
    int filter_count = 0;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            filter_count += mean_filter.at<uchar>(i, j); //����ģ��ĺ�
        }
    }
    for (int i = 0; i < rownum - x; i++)
    {
        for (int j = 0; j < colnum - y; j++)
        {
            temp = 0;
            for (int k = 0; k < x; k++) //����������ֵ
            {
                for (int m = 0; m < y; m++)
                {
                    temp = temp + (src.at<uchar>(i + k, j + m) * mean_filter.at<uchar>(k, m)); //���о�������
                }
            }
            temp = temp / filter_count;
            image.at<uchar>(i + (x - 1) / 2, j + (y - 1) / 2) = temp;
        }
    }
    namedWindow("均值变换", WINDOW_AUTOSIZE);
    imshow("均值变换", image);
    waitKey();
}

int main()
{
    Meanfilter_Grey(5, 5);
    Meanfilter_Color(5, 5);
    Gaussianfilter_Gray(5, 5);
    Gaussianfilter_Color(9, 9);
    Laplacianfilter_Gray(); //针对单一的拉普拉斯算子模板进行变换0-10,-14-10-10
    Laplacianfilter_Color();
    Robertfilter_Gray();
    Robertfilter_Color();
    Sobelfilter_Gray();
    Sobelfilter_Color();
    Highfilter(3, 3);
}