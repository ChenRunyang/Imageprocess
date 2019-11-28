#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class HistogramND
{
private:
    Mat image;                           //‘¥ÕºœÒ
    int hisSize[1], hisWidth, hisHeight; //÷±∑ΩÕºµƒ¥Û–°,øÌ∂»∫Õ∏ﬂ∂»
    float range[2];                      //÷±∑ΩÕº»°÷µ∑∂Œß
    const float *ranges;
    Mat channelsRGB[3]; //∑÷¿ÎµƒBGRÕ®µ¿
    MatND outputRGB[3]; // ‰≥ˆ÷±∑ΩÕº∑÷¡ø
public:
    HistogramND()
    {
        hisSize[0] = 256;
        hisWidth = 400;
        hisHeight = 400;
        range[0] = 0.0;
        range[1] = 256.0;
        ranges = &range[0];
    }

    //µº»ÎÕº∆¨
    bool importImage(String path)
    {
        image = imread(path);
        if (!image.data)
            return false;
        return true;
    }

    //∑÷¿ÎÕ®µ¿
    void splitChannels()
    {
        split(image, channelsRGB);
    }

    //º∆À„÷±∑ΩÕº
    void getHistogram()
    {
        calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
        calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
        calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);

        // ‰≥ˆ∏˜∏ˆbinµƒ÷µ
        for (int i = 0; i < hisSize[0]; ++i)
        {
            cout << i << "   B:" << outputRGB[0].at<float>(i);
            cout << "   G:" << outputRGB[1].at<float>(i);
            cout << "   R:" << outputRGB[2].at<float>(i) << endl;
        }
    }

    //œ‘ æ÷±∑ΩÕº
    void displayHisttogram()
    {
        Mat rgbHist[3];
        for (int i = 0; i < 3; i++)
        {
            rgbHist[i] = Mat(hisWidth, hisHeight, CV_8UC3, Scalar::all(0));
        }
        normalize(outputRGB[0], outputRGB[0], 0, hisWidth - 20, NORM_MINMAX);
        normalize(outputRGB[1], outputRGB[1], 0, hisWidth - 20, NORM_MINMAX);
        normalize(outputRGB[2], outputRGB[2], 0, hisWidth - 20, NORM_MINMAX);
        for (int i = 0; i < hisSize[0]; i++)
        {
            int val = saturate_cast<int>(outputRGB[0].at<float>(i));
            rectangle(rgbHist[0], Point(i * 2 + 10, rgbHist[0].rows),
                      Point((i + 1) * 2 + 10, rgbHist[0].rows - val), Scalar(0, 0, 255), 1, 8);
            val = saturate_cast<int>(outputRGB[1].at<float>(i));
            rectangle(rgbHist[1], Point(i * 2 + 10, rgbHist[1].rows), Point((i + 1) * 2 + 10, rgbHist[1].rows - val), Scalar(0, 255, 0), 1, 8);
            val = saturate_cast<int>(outputRGB[2].at<float>(i));
            rectangle(rgbHist[2], Point(i * 2 + 10, rgbHist[2].rows), Point((i + 1) * 2 + 10, rgbHist[2].rows - val), Scalar(255, 0, 0), 1, 8);
        }

        cv::imshow("R", rgbHist[0]);
        imshow("G", rgbHist[1]);
        imshow("B", rgbHist[2]);
        imshow("image", image);
    }

    void equallize(HistogramND &img1)
    {
        int bmap[256];
        float pr[256];
        float pre[257];
        int dmap[256];
        float temp;
        pre[0] = 0;
        for (int k = 0; k < 3; k++)
        {
            for (int i = 0; i < 256; i++)
            {
                bmap[i] = 0;
            }
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    bmap[image.at<Vec3b>(i, j)[k]]++;
                }
            }
            for (int i = 0; i < 256; i++)
            {
                pr[i] = (float)bmap[i] / (float)(image.rows * image.cols);
            }
            for (int i = 0; i < 256; i++)
            {
                temp = 255 * pr[i] + pre[i];
                pre[i + 1] = temp;
                dmap[i] = (int)(temp + 0.5);
            }
            for (int i = 0; i < img1.image.rows; i++)
            {
                for (int j = 0; j < img1.image.cols; j++)
                {
                    img1.image.at<Vec3b>(i, j)[k] = dmap[image.at<Vec3b>(i, j)[k]];
                }
            }
        }
    }
};

class HistogramOD
{
private:
    MatND channel;
    MatND outchannel;
    int width, height, size[1];
    float range[2];
    const float *ranges;

public:
    Mat image;
    HistogramOD()
    {
        size[0] = 256;
        width = 400;
        height = 400;
        range[0] = 0.0;
        range[1] = 256.0;
        ranges = &range[0];
    }

    bool importImage(String path)
    {
        image = imread(path, 0);
        if (!image.data)
            return false;
        else
            return true;
    }

    void getHistogram()
    {
        calcHist(&image, 1, 0, Mat(), outchannel, 1, size, &ranges);
    }

    void displayHisttogram()
    {
        Mat hist(width, height, CV_8UC1, Scalar::all(0));
        normalize(outchannel, outchannel, 0, width - 20, NORM_MINMAX);
        for (int i = 0; i < size[0]; i++)
        {
            int val = saturate_cast<int>(outchannel.at<float>(i));
            rectangle(hist, Point(i * 2 + 10, hist.rows),
                      Point((i + 1) * 2 + 10, hist.rows - val), Scalar(255), 1, 8);
        }
        cv::imshow("G", hist);
        imshow("image", image);
    }

    void equallize(HistogramOD &img1)
    {
        int bmap[256];
        float pr[256];
        float pre[257];
        int dmap[256];
        float temp;
        pre[0] = 0;
        for (int i = 0; i < 256; i++)
        {
            bmap[i] = 0;
        }
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                bmap[image.at<uchar>(i, j)]++;
            }
        }
        for (int i = 0; i < 256; i++)
        {
            pr[i] = (float)bmap[i] / (float)(image.rows * image.cols);
        }
        for (int i = 0; i < 256; i++)
        {
            temp = 255 * pr[i] + pre[i];
            pre[i + 1] = temp;
            dmap[i] = (int)(temp + 0.5);
        }
        for (int i = 0; i < img1.image.rows; i++)
        {
            for (int j = 0; j < img1.image.cols; j++)
            {
                img1.image.at<uchar>(i, j) = dmap[image.at<uchar>(i, j)];
            }
        }
        normalize(img1.image, img1.image, 0, 255, CV_MINMAX);
    }
};

int main(int argc, char *argv[])
{
    String path = "demo.jpg";
    /*HistogramOD hist;                               //µ•Õ®µ¿±‰ªª
	HistogramOD det;
	if (!hist.importImage(path)){
		cout << "Import Error!" << endl;
		return -1;
	}
	hist.getHistogram();
	hist.displayHisttogram();
	cv::waitKey();
	if (!det.importImage(path)){
		cout << "Import Error!" << endl;
		return -1;
	}
	hist.equallize(det);
	det.getHistogram();
	det.displayHisttogram();
	cv::waitKey();
	*/
    HistogramND hist;
    HistogramND det;
    if (!hist.importImage(path))
    {
        cout << "Import Error!" << endl;
        return -1;
    }
    hist.splitChannels();
    hist.getHistogram();
    hist.displayHisttogram();
    cv::waitKey();
    if (!det.importImage(path))
    {
        cout << "Import Error!" << endl;
        return -1;
    }
    hist.equallize(det);
    det.splitChannels();
    det.getHistogram();
    det.displayHisttogram();
    cv::waitKey();

    return 0;
}