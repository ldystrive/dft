#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "dft.h"

using namespace std;

cv::Mat idft_opencv;
cv::Mat idft_ldy;

cv::Mat convertFreq(const cv::Mat real, const cv::Mat imag)
{
	cv::Mat img;
	cv::magnitude(real, imag, img);
	cv::Mat magI = img;
	magI += cv::Scalar::all(1);
	cv::log(magI, magI);
	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));

	cv::Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);
	return magI;
}

void opencv_test(cv::Mat img)
{
	int M = cv::getOptimalDFTSize(img.rows);
	int N = cv::getOptimalDFTSize(img.cols);
	
	cv::Mat padded;
	cv::copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexImg;
	cv::merge(planes, 2, complexImg);
	
	cv::dft(complexImg, complexImg);
	cv::split(complexImg, planes);
	
	cv::Mat magI = convertFreq(planes[0], planes[1]);
	cv::imshow("dft", magI);

	cv::idft(complexImg, complexImg, cv::DFT_SCALE);
	cv::split(complexImg, planes);
	planes[0].copyTo(idft_opencv);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	cv::imshow("idft", planes[0]);
}

void ldy_test(const float *img, int height, int width)
{
	int M = dft::getOptimalDFTSize(height);
	int N = dft::getOptimalDFTSize(width);
	float *src0 = new float[M * N];
	float *src  = new float[M * N * 2];
	float *dst_dft  = new float[M * N * 2];
	float *dst_idft = new float[M * N * 2];
	dft::copyMakeBorder(img, src0, height, width, M, N);

	// convert to 2 channels
	for (int i = 0; i < M * N; i++) {
		src[i << 1] = src0[i];
		src[i << 1 | 1] = 0;
	}

	dft::dft(src, dst_dft, M, N);
	dft::idft(dst_dft, dst_idft, M, N);
	
	/********************************************************/
	cv::Mat complexImg(M, N, CV_32FC2, dst_dft);
	cv::Mat planes[2];
	cv::split(complexImg, planes);

	cv::Mat magI = convertFreq(planes[0], planes[1]);
	
	cv::imshow("dft_ldy", magI);
	
	cv::Mat complexImg_idft(M, N, CV_32FC2, dst_idft);
	cv::split(complexImg_idft, planes);
	planes[0].copyTo(idft_ldy);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	cv::imshow("idft_ldy", planes[0]);
	
	free(src0);
	free(src);
	free(dst_dft);
	free(dst_idft);
}

int main()
{
	string img_path("testImg/img.PNG");
	cv::Mat img = cv::imread(img_path);
	cv::Mat imgF;
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(imgF, CV_32F);
	cv::imshow("src", img);

	opencv_test(imgF);
	ldy_test(imgF.ptr<float>(), img.rows, img.cols);

	cv::waitKey(0);

	return 0;
}