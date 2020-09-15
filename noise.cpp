#include <iostream>
#include <string>
#include <cstdlib>
#include <limits>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

using namespace std;
using namespace cv;

//加入椒盐噪声
void addSalt(Mat &image, int n)
{
	int i, j;
	for (int k = 0; k < n; k++) //将n个像素随机置0
	{
		i = rand() % image.cols;
		j = rand() % image.rows;
		//将颜色随机改变
		if (image.channels() == 1)
			image.at<uchar>(j, i) = 255;
		else
		{
			for (int t = 0; t < image.channels(); t++)
			{
				image.at<Vec3b>(j, i)[t] = 255;
			}
		}
	}
}

void addPepper(Mat &image, int n) //加入椒噪声
{
	for (int k = 0; k < n; k++) //将n个像素随机置0
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		//将像素随机改变
		if (image.channels() == 1)
			image.at<uchar>(j, i) = 0;
		else
		{
			for (int t = 0; t < image.channels(); t++)
			{
				image.at<Vec3b>(j, i)[t] = 0;
			}
		}
	}
}

int GaussianNoise(double mu, double sigma)
{
	//定义一个极小量
	const double epsilon = numeric_limits<double>::min(); //返回目标数据类型表示最接近1的正数和1的差的绝对值
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假，返回随机变量
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;

	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真，构造随机变量
	z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
	return z1 * sigma + mu;
}

Mat addGaussianNoise(Mat &srcImage)
{
	Mat resultImage = srcImage.clone();
	int channels = resultImage.channels(); //获取图像通道
	int nRows = resultImage.rows;		   //获取图像行数

	int nCols = resultImage.cols * channels; //获取图像列数
	//判断连续性
	if (resultImage.isContinuous()) //若连续，只需要遍历一维数组
	{
		nCols *= nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{ //���Ӹ�˹����
			int val = resultImage.ptr<uchar>(i)[j] + GaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			resultImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return resultImage;
}
//中值滤波器
void medeanFilter(Mat &src, int win_size)
{
	int rows = src.rows, cols = src.cols;
	int start = win_size / 2;
	for (int m = start; m < rows - start; m++)
	{
		for (int n = start; n < cols - start; n++)
		{
			vector<uchar> model;
			for (int i = -start + m; i <= start + m; i++)
			{
				for (int j = -start + n; j <= start + n; j++)
				{
					model.push_back(src.at<uchar>(i, j));
				}
			}
			sort(model.begin(), model.end());
			src.at<uchar>(m, n) = model[win_size * win_size / 2];
		}
	}
}

//均值滤波器
void meanFilter(Mat &src, int win_size)
{
	int rows = src.rows, cols = src.cols;
	int start = win_size / 2;
	for (int m = start; m < rows - start; m++)
	{
		for (int n = start; n < cols - start; n++)
		{
			if (src.channels() == 1) //灰度图
			{
				int sum = 0;
				for (int i = -start + m; i <= start + m; i++)
				{
					for (int j = -start + n; j <= start + n; j++)
					{
						sum += src.at<uchar>(i, j);
					}
				}
				src.at<uchar>(m, n) = uchar(sum / win_size / win_size);
			}
			else
			{
				Vec3b pixel;
				int sum1[3] = {0};
				for (int i = -start + m; i <= start + m; i++)
				{
					for (int j = -start + n; j <= start + n; j++)
					{
						pixel = src.at<Vec3b>(i, j);
						for (int k = 0; k < src.channels(); k++)
						{
							sum1[k] += pixel[k];
						}
					}
				}
				for (int k = 0; k < src.channels(); k++)
				{
					pixel[k] = sum1[k] / win_size / win_size;
				}
				src.at<Vec3b>(m, n) = pixel;
			}
		}
	}
}

//几何均值滤波器
Mat GeometryMeanFilter(Mat src)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double mul;
	double dc;
	int mn;
	//计算去燥的color值
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{

			if (src.channels() == 1) //灰度图
			{
				mul = 1.0;
				mn = 0;
				//计算几何均值，领域大小5*5
				for (int m = -2; m <= 2; m++)
				{
					row = i + m;
					for (int n = -2; n <= 2; n++)
					{
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w)
						{
							int s = src.at<uchar>(row, col);
							mul = mul * (s == 0 ? 1 : s); //非零节点相乘，最小值为1
							mn++;
						}
					}
				}
				//计算1/mn次方
				dc = pow(mul, 1.0 / mn);
				//对图像进行统计
				int res = (int)dc;
				dst.at<uchar>(i, j) = res;
			}
			else
			{
				double multi[3] = {1.0, 1.0, 1.0};
				mn = 0;
				Vec3b pixel;

				for (int m = -2; m <= 2; m++)
				{
					row = i + m;
					for (int n = -2; n <= 2; n++)
					{
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w)
						{
							pixel = src.at<Vec3b>(row, col);
							for (int k = 0; k < src.channels(); k++)
							{
								multi[k] = multi[k] * (pixel[k] == 0 ? 1 : pixel[k]); //非零节点相乘，最小值为1
							}
							mn++;
						}
					}
				}
				double d;
				for (int k = 0; k < src.channels(); k++)
				{
					d = pow(multi[k], 1.0 / mn);
					pixel[k] = (int)d;
				}
				dst.at<Vec3b>(i, j) = pixel;
			}
		}
	}
	return dst;
}

//谐波均值滤波器，模版5*5
Mat HarmonicMeanFilter(Mat src)
{
	//IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double sum;
	double dc;
	int mn;
	int mulnum = 1;
	//计算去燥后的color值
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			sum = 0.0;
			mn = 0;
			//统计领域
			for (int m = -2; m <= 2; m++)
			{
				row = i + m;
				for (int n = -2; n <= 2; n++)
				{
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w)
					{
						int s = src.at<uchar>(row, col);
						sum = sum + 1.0 / (s == 0 ? 255 : s); //0设置为255
						mn++;
					}
				}
			}
			int d;
			dc = mn / sum;
			d = dc;
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

//逆谐波均值大小滤波器 模版大小5*5
Mat InverseHarmonicMeanFilter(Mat src, double Q)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double sum;
	double sum1;
	double dc;
	//计算去燥后的color值 ֵ
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			sum = 0.0;
			sum1 = 0.0;
			//统计领域
			for (int m = -2; m <= 2; m++)
			{
				row = i + m;
				for (int n = -2; n <= 2; n++)
				{
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w)
					{

						int s = src.at<uchar>(row, col);
						sum = sum + pow(s, Q + 1);
						sum1 = sum1 + pow(s, Q);
					}
				}
			}
			//计算1/mn
			int d;
			dc = sum1 == 0 ? 0 : (sum / sum1);
			d = (int)dc;
			//赋给去燥后的图像
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

//自适应均值滤波
Mat SelfAdaptMedianFilter(Mat src)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double Zmin, Zmax, Zmed, Zxy, Smax = 7;
	int wsize;
	//计算去燥后的color值
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			//统计领域
			wsize = 1;
			while (wsize <= 3)
			{
				Zmin = 255.0;
				Zmax = 0.0;
				Zmed = 0.0;
				int Zxy = src.at<uchar>(i, j);
				int mn = 0;
				for (int m = -wsize; m <= wsize; m++)
				{
					row = i + m;
					for (int n = -wsize; n <= wsize; n++)
					{
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w)
						{
							int s = src.at<uchar>(row, col);
							if (s > Zmax)
							{
								Zmax = s;
							}
							if (s < Zmin)
							{
								Zmin = s;
							}
							Zmed = Zmed + s;
							mn++;
						}
					}
				}
				Zmed = Zmed / mn;
				int d;
				if ((Zmed - Zmin) > 0 && (Zmed - Zmax) < 0)
				{
					if ((Zxy - Zmin) > 0 && (Zxy - Zmax) < 0)
					{
						d = Zxy;
					}
					else
					{
						d = Zmed;
					}
					dst.at<uchar>(i, j) = d;
					break;
				}
				else
				{
					wsize++;
					if (wsize > 3)
					{
						int d;
						d = Zmed;
						dst.at<uchar>(i, j) = d;
						break;
					}
				}
			}
		}
	}
	return dst;
}

//自适应均值滤波
Mat SelfAdaptMeanFilter(Mat src)
{
	Mat dst = src.clone();
	blur(src, dst, Size(7, 7));
	int row, col;
	int h = src.rows;
	int w = src.cols;
	int mn;
	double Zxy;
	double Zmed;
	double Sxy;
	double Sl;
	double Sn = 100;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int Zxy = src.at<uchar>(i, j);
			int Zmed = src.at<uchar>(i, j);
			Sl = 0;
			mn = 0;
			for (int m = -3; m <= 3; m++)
			{
				row = i + m;
				for (int n = -3; n <= 3; n++)
				{
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w)
					{
						int Sxy = src.at<uchar>(row, col);
						Sl = Sl + pow(Sxy - Zmed, 2);
						mn++;
					}
				}
			}
			Sl = Sl / mn;
			int d = (int)(Zxy - Sn / Sl * (Zxy - Zmed));
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

IplImage *MatToIplImage(Mat image)
{
	Mat t = image.clone();
	IplImage *res = &IplImage(t);
	return res;
}

Mat IplImageToMat(IplImage *image)
{
	Mat res = cvarrToMat(image, true);
	return res;
}

void test1()
{
	Mat image, noise, res;

	/*----------高斯噪声 算术均值-----------*/
	image = imread("demo.jpg", 0);
	imshow("原图", image);

	noise = addGaussianNoise(image); //添加噪声
	imshow("高斯噪声", noise);

	res = noise.clone();
	meanFilter(res, 5); //算术均值滤波
	imshow("算术均值滤波器", res);
	waitKey(0);
	destroyAllWindows();

	/*----------胡椒噪声 几何均值-----------*/
	image = imread("demo.jpg", 0); // Read the file
	imshow("原图", image);		   // Show our image inside it.

	noise = image.clone();
	addPepper(noise, 1000);
	imshow("添加了1000个胡椒噪声", noise);

	res = noise.clone();
	meanFilter(res, 5);
	imshow("几何均值滤波器", res);

	waitKey(0);
	destroyAllWindows();

	/*--------------椒盐噪声 逆均值滤波器------------*/
	image = imread("demo.jpg", 0); // Read the file
	imshow("原图", image);		   // Show our image inside it.

	noise = image.clone();
	addSalt(noise, 1000);
	imshow("1000个盐噪声", noise);

	res = HarmonicMeanFilter(noise);
	imshow("5*5г����ֵ�˲���", res);

	/*------չʾͼ��-------*/
	waitKey(0);
	destroyAllWindows();

	/*-----------��������+��г����ֵ�˲���-----------*/
	image = imread("demo.jpg", 0);
	imshow("ԭʼͼ��", image);

	noise = image.clone();
	addSalt(noise, 1000); //��ֹ���������һ��
	addPepper(noise, 1000);
	imshow("����1000��������+1000����������", noise);

	res = InverseHarmonicMeanFilter(noise, 1); //�ڶ���������Q��Q=0�˻���������ֵ
	imshow("5*5��г����ֵ�˲���", res);

	/*------չʾͼ��-------*/
	waitKey(0);
	destroyAllWindows();
	return;
}

void test2()
{
	Mat image, noise, res1, res2;

	/*---------����------------*/
	image = imread("demo.jpg", 0);
	imshow("ԭʼͼ��", image);

	noise = image.clone();
	addPepper(noise, 1000);
	imshow("����1000����������", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5�о�ֵ�˲���", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9�о�ֵ�˲���", res2);
	/*------չʾͼ��-------*/
	waitKey(0);
	destroyAllWindows();

	/*-----------������---------------*/
	image = imread("demo.jpg", 0);
	imshow("ԭʼͼ��", image);

	noise = image.clone();
	addSalt(noise, 1000);
	imshow("����1000��������", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5�о�ֵ�˲���", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9�о�ֵ�˲���", res2);
	/*------չʾͼ��-------*/
	waitKey(0);
	destroyAllWindows();

	/*-----------������+��������---------------*/
	image = imread("demo.jpg", 0);
	imshow("ԭʼͼ��", image);

	noise = image.clone();
	addSalt(noise, 1000);
	addPepper(noise, 1000);
	imshow("����1000��������+1000����������", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5�о�ֵ�˲���", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9�о�ֵ�˲���", res2);
	/*------չʾͼ��-------*/
	waitKey(0);
	destroyAllWindows();
	return;
}

void test3()
{
	Mat image, res1, res2, noise;
	image = imread("demo.jpg", 0); // Read the file
	imshow("原图", image);

	noise = image.clone();
	addPepper(noise, 1000);
	addSalt(noise, 1000);
	imshow("1000椒盐", noise);

	res1 = SelfAdaptMeanFilter(image);
	imshow("自适应均值滤波", res1);

	res2 = noise.clone();
	meanFilter(res2, 7);
	imshow("7*7均值滤波", res2);

	waitKey(0);
	destroyAllWindows();
}

void test4()
{
	Mat image, res1, res2, noise;
	image = imread("demo.jpg", 0);
	imshow("原图", image);

	noise = image.clone();
	addPepper(noise, 1000);
	addSalt(noise, 1000);
	imshow("1000个椒盐噪声", noise);

	res1 = SelfAdaptMedianFilter(image);
	imshow("自适应中值滤波", res1);

	res2 = noise.clone();
	medeanFilter(res2, 7);
	imshow("7*7中值滤波", res2);

	waitKey(0);
	destroyAllWindows();
}

void test5()
{
	Mat image, res1, res2, noise;
	image = imread("demo.jpg", 1);
	imshow("原图", image);

	noise = addGaussianNoise(image);
	imshow("添加高斯噪声", noise);

	res1 = noise.clone();
	meanFilter(res1, 5);
	imshow("中指滤波", res1);

	res2 = GeometryMeanFilter(noise);
	imshow("���ξ�ֵ�˲���", res2);

	waitKey(0);
	destroyAllWindows();
}
int forth_main()
{
	test1();
	test2();
	test3();
	test4();
	test5();
	destroyAllWindows();
	return 0;
}