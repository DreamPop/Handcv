#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <cstring>
#include <vector>

#define thresholds	0.05

using namespace cv;
using namespace std;


/*
//提取肤色，skinarea是二值化的图像
void SkinFind(const Mat& src, Mat& skinarea) {
	Mat YCbCr;	//YCbCr色度空间用于识别皮肤效果会好些
	vector<Mat> p;
	//将RGB图转成YCbCr图
	cvtColor(src, YCbCr, CV_BGR2YCrCb);
	//把YCrCb图分成chennal个矩阵
	split(YCbCr, p);

	MatIterator_<uchar> itCb = p[1].begin<uchar>();
	MatIterator_<uchar> itCb_END = p[1].end<uchar>();
	MatIterator_<uchar> itCr = p[2].begin<uchar>();
	MatIterator_<uchar> its = skinarea.begin<uchar>();

	//是人的皮肤颜色在YCbCr色. 度空间的分布范围。 100≤ Cb≤ 127， 138≤ Cr≤ 170.
	//file.lw23.com/8/8c/8c1/8c145911-8e1d-4798-97b0-12b1693d0085.pdf‎file.lw23.com/8/8c/8c1/8c145911-8e1d-4798-97b0-12b1693d0085.pdf‎

	for( ; itCb != itCb_END; ++itCr, ++itCb, ++its) {
		if(*itCb >= 133 && *itCb <= 173 && *itCr >= 77 && *itCr <= 127){
			*its = 255;
		} else {
			*its = 0;
		}
	}

	// 形态学操作，去除噪声，并使手的边界更加清晰
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
	erode(skinarea, skinarea, element);
	morphologyEx(skinarea, skinarea, MORPH_OPEN, element);
	dilate(skinarea, skinarea, element);
	morphologyEx(skinarea, skinarea, MORPH_CLOSE, element);
}

//给图像做光线补偿
void LightImage(IplImage * read) {
	IplImage *img = cvCreateImage(cvGetSize(read), 8, 1);
	IplImage *r = cvCreateImage(cvGetSize(read), 8, 1);
	IplImage *g = cvCreateImage(cvGetSize(read), 8, 1);
	IplImage *b = cvCreateImage(cvGetSize(read), 8, 1);

	cvSplit(read, b, g, r, 0);
	cvCvtColor(read, img, CV_BGR2GRAY);
	int* gray = new int[256];
	int i, j, w, h, tol;
	
	w = read->width;
	h = read->height;
	tol = w*h;
	memset(gray, 0, sizeof(int)*256);

	for(i = 0; i < h; ++i) {
		for(j = 0; j < w; ++j) {
			gray[((uchar*)(img->imageData + i*img->width))[j]]++;	
		}
	}

	int cal = 0, p;
	for(i = 0; i < 256; ++i) {
		if(double(cal)/tol < thresholds) {	//得到前5%的高亮像素
			cal += gray[255-i];
			p = i;
		} else	break;
	}
	int avg = 0;
	for(i = 255; i >= 255 - p; --i) {
		avg += gray[i]*i;
	}
	avg /= cal;
	double fac = 255./double(avg);
	
	for(i = 0; i < h; ++i) {
		for(j = 0; j < w; ++j) {
			int bb = ((uchar*)(b->imageData + i*b->widthStep))[j];
			bb = bb*fac > 255 ? 255 : bb*fac;
			((uchar*)(b->imageData + i*b->widthStep))[j] = bb;

			int gg = ((uchar*)(g->imageData + i*g->widthStep))[j + 1];
			gg = gg*fac > 255 ? 255 : gg*fac;
			((uchar*)(g->imageData + i*g->widthStep))[j + 1] = gg;

			int rr = ((uchar*)(r->imageData + i*r->widthStep))[j + 2];
			rr = rr*fac > 255 ? 255 : rr*fac;
			((uchar*)(r->imageData + i*r->widthStep))[j + 2] = rr;			
		}
	}
	cvMerge(b, g, r, 0, read);
	cvReleaseImage(&r);
	cvReleaseImage(&g);
	cvReleaseImage(&b);
	cvReleaseImage(&img);
	delete[] gray;
}

int main() {
	CvCapture* capture = cvCaptureFromCAM(0);
	Mat src, skinarea;
	IplImage* read = 0;
	read = cvQueryFrame(capture);
	if(!read) {
		printf("Cannot open Camera!");
		return -1;
	}

	vector<vector<Point>> contours;	//轮廓
	vector<Vec4i> hierarchy;	//分层，每一个轮廓对应四个herarchy.
	vector<Point> couP;

	while(1) {
		read = cvQueryFrame(capture);
		if(!read)	break;
		LightImage(read);
		src = Mat(read, false);

		imshow("show", src);

		medianBlur(src, src, 5);
		GaussianBlur(src, src, Size(3, 3), 0);	//高斯滤波，边缘提升，其他地方虚化
		imshow("After Gauss", src);
		skinarea.create(src.rows, src.cols, CV_8UC1);
		SkinFind(src, skinarea);	//找到手势区域
		Mat outimg;
		src.copyTo(outimg, skinarea);	//src 复制到outimg，skinarea做过滤，当skinarea像素为0时不复制，当skinarea像素不为0时复制

		contours.clear();
		hierarchy.clear();
		findContours(skinarea, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		int idx = 0;
		double area, maxa = -1;
		for(int i = 0; i < (int)contours.size(); ++i) {
			area = contourArea(Mat(contours[i]));
			if(area > maxa) {
				idx = i;
				maxa = area;
			}
		}
		
		if(maxa == -1)	{
			imshow("Output Image", outimg);
			continue;
		}
		couP.clear();  
		for(int i = 0; i < (int)contours[idx].size(); ++i) {
			couP.push_back(contours[idx][i]);
		}
		//vector<Point> numP;
		Point a, b, c;
		for(int i = 10; couP.size() && (i < (int)couP.size() - 10); i += 10) {
			a = couP[i - 10];
			b = couP[i];
			c = couP[i + 10];
			int dot = (a.x - b.x) * (a.y - b.y) + (c.x - b.x) * (c.y - b.y);	//点积判夹角
			if(dot > 80 || dot < -80) {	
				int cross = (a.x - b.x)*(c.y - b.y) - (a.y - b.y)*(c.x - b.x);	//差积判方向
				if(cross < 0) {	//右手定则
					//snumP.push_back(b);
					circle(outimg, b, 5, Scalar(255, 0, 0), CV_FILLED);
				}
			}
			
		}
		imshow("Output Image", outimg);
		outimg.release();

		if(cvWaitKey(30) == 'q') {
			break;
		}
	}
	return 0;
}
*/

CascadeClassifier fist_cascade;
CascadeClassifier palm_cascade;

void decectAndDisplay(Mat frame) {
	vector<Rect> fists;
	vector<Rect> palms;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);	//直方图均衡化
	fist_cascade.detectMultiScale(frame_gray, fists, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for(int i = 0; i < fists.size(); ++i) {
		Point center(fists[i].x + fists[i].width*0.5, fists[i].y + fists[i].height*0.5);
		ellipse(frame, center, Size(fists[i].width*0.5, fists[i].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
		//circle(frame, center, (fists[i].width + fists[i].width)*0.5, Scalar(0, 0, 255), 1, 8, 0);
	}

	palm_cascade.detectMultiScale(frame_gray, palms, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for(int i = 0; i < palms.size(); ++i) {
		Point center(palms[i].x + palms[i].width*0.5, palms[i].y + palms[i].height*0.5);
		ellipse(frame, center, Size(palms[i].width*0.5, palms[i].height*0.5), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
		//circle(frame, center, (palms[i].width + palms[i].width)*0.5, Scalar(255, 0, 0), 1, 8, 0);
	}
	imshow("Detect", frame);
}

int main(int argc, const char** argv) {
	CvCapture* capture;
	Mat src;
	if(!fist_cascade.load("fist.dat")) {printf("Load fist error\n"); return -1;}
	if(!palm_cascade.load("palm.dat")) {printf("Load palm error\n"); return -1;}
	capture = cvCaptureFromCAM(-1);
	if(!capture)	{printf("Camera not find!\n"); return -1;}
	while(true) {
		src = cvQueryFrame(capture);
		if(src.empty())	{printf("Error 01\n"); break;}
		decectAndDisplay(src);
		if(waitKey(10) == 'q')	return 0;
	}
	return 0;
} 