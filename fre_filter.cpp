
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

Mat DFT(Mat I)
{
	Mat padded;
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);

	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);

	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];

	magI += Scalar::all(1);
	log(magI, magI);

	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	//重新排列傅立叶图像的象限，使其位于图像中心
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));	//左上角
	Mat q1(magI, Rect(cx, 0, cx, cy));	//右上角
	Mat q2(magI, Rect(0, cy, cx, cy));	//左下角
	Mat q3(magI, Rect(cx, cy, cx, cy)); //右下角

	//变换左上角和右下角
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	//变换右上和左下
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//归一化处理
	normalize(magI, magI, 0, 1, 32);

	return magI;
}

void test7()
{
	Mat image, res;
	image = imread("demo.jpg", 0); // Read the file
	imshow("原图", image);

	res = DFT(image);
	imshow("频谱图", res);
	waitKey(0);
	destroyAllWindows();
	return;
}

int DFTAndIDFT()
{
	Mat input = imread("demo.jpg", 0);
	imshow("input", input);
	int w = getOptimalDFTSize(input.cols);
	int h = getOptimalDFTSize(input.rows);
	Mat padded;
	copyMakeBorder(input, padded, 0, h - input.rows, 0, w - input.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat plane[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexIm;
	merge(plane, 2, complexIm);
	dft(complexIm, complexIm);
	split(complexIm, plane);
	magnitude(plane[0], plane[1], plane[0]);
	int cx = padded.cols / 2;
	int cy = padded.rows / 2;
	Mat temp;
	Mat part1(plane[0], Rect(0, 0, cx, cy));
	Mat part2(plane[0], Rect(cx, 0, cx, cy));
	Mat part3(plane[0], Rect(0, cy, cx, cy));
	Mat part4(plane[0], Rect(cx, cy, cx, cy));
	part1.copyTo(temp);
	part4.copyTo(part1);
	temp.copyTo(part4);
	part2.copyTo(temp);
	part3.copyTo(part2);
	temp.copyTo(part3);

	Mat _complexim;
	complexIm.copyTo(_complexim);
	Mat iDft[] = {Mat::zeros(plane[0].size(), CV_32F), Mat::zeros(plane[0].size(), CV_32F)};
	idft(_complexim, _complexim);
	split(_complexim, iDft);
	magnitude(iDft[0], iDft[1], iDft[0]);
	normalize(iDft[0], iDft[0], 1, 0, CV_MINMAX);
	imshow("idft", iDft[0]);
	plane[0] += Scalar::all(1);
	log(plane[0], plane[0]);
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);
	imshow("dft", plane[0]);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

void ideal_Low_Pass_Filter(double D0 = 60)
{
	Mat src, fourier, res;
	src = imread("demo.jpg", 0); // Read the file
	imshow("原图", src);
	Mat img = src.clone();
	//调整图像大小
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//记录实部和虚部
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);
	//傅立叶变换
	dft(complexImg, complexImg);
	//获取图像
	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	//��ʵ��Ϊ�˰��к��б��ż�� -2�Ķ�������11111111.......10 ���һλ��0
	//获取中心点
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//调整频域
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//按公式保留中心位置
	for (int y = 0; y < mag.rows; y++)
	{
		double *data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0)
			{
			}
			else
			{
				data[x] = 0;
			}
		}
	}
	//调整频域
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("低通滤波器", invDFTcvt);
	waitKey(0);
	destroyAllWindows();
	return;
}

void ideal_High_Pass_Filter(double D0 = 60)
{
	Mat src, fourier, res;
	src = imread("demo.jpg", 0); // Read the file
	imshow("原图", src);
	Mat img = src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	for (int y = 0; y < mag.rows; y++)
	{
		double *data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0)
			{
				data[x] = 0;
			}
			else
			{
			}
		}
	}
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("�����ͨ�˲���", invDFTcvt);
	waitKey(0);
	destroyAllWindows();
	return;
}

void Butterworth_Low_Paass_Filter(double D0 = 60, int n = 2)
{
	Mat src, fourier, res;
	src = imread("demo.jpg", 0); // Read the file
	imshow("原图", src);

	//H = 1 / (1+(D/D0)^2n)
	Mat img = src.clone();
	//cvtColor(src, img, CV_BGR2GRAY);
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	for (int y = 0; y < mag.rows; y++)
	{
		double *data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{
			//cout << data[x] << endl;
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			//cout << d << endl;
			double h = 1.0 / (1 + pow(d / D0, 2 * n));
			if (h <= 0.5)
			{
				data[x] = 0;
			}
			else
			{
				//data[x] = data[x]*0.5;
				//cout << h << endl;
			}

			//cout << data[x] << endl;
		}
	}
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//��任
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("巴特沃斯高通滤波器", invDFTcvt);

	waitKey(0);
	destroyAllWindows();
	return;
}

void Butterworth_High_Paass_Filter(double D0 = 60, int n = 2)
{
	Mat src, fourier, res;
	src = imread("demo.jpg", 0); // Read the file
	imshow("原图", src);

	Mat img = src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	for (int y = 0; y < mag.rows; y++)
	{
		double *data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{

			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));

			double h = 1.0 / (1 + pow(D0 / d, 2 * n));
			if (h <= 0.5)
			{
				data[x] = 0;
			}
			else
			{
			}
		}
	}
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("巴特沃斯高通滤波器", invDFTcvt);

	waitKey(0);
	destroyAllWindows();
	return;
}

int fifth_main()
{
	DFTAndIDFT();
	ideal_Low_Pass_Filter(40.0);
	ideal_High_Pass_Filter(40.0);
	Butterworth_Low_Paass_Filter(40, 2);
	Butterworth_High_Paass_Filter(40, 2);

	return 0;
}