
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "corner.h"
#include "fast_cuda.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"


int main( void )
{
	using namespace std;
	using namespace cv;

    const int threshold=50;
    string filename1="/Users/macbookpro/Downloads/lena.jpg";
    string filename2="/Users/macbookpro/Downloads/lena_rotate.jpg";
    const Mat img1 =imread(filename1,0);   // Read the file
    const Mat img2 =imread(filename2,0);
    if(! (img1.data&&img2.data) )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	fast_corner_9(img1,keypoint1,threshold);
	fast_corner_9(img2,keypoint2,threshold);

	Mat image1=imread(filename1,1);
	Mat image2=imread(filename2,1);
	drawKeypoints(image1,keypoint1,image1,Scalar(0,255,0),0);
	drawKeypoints(image2,keypoint2,image2,Scalar(0,255,0),0);
	Size size1=img1.size();
	Size size2=img2.size();
	Mat img_combine(img1.rows,img1.cols+img2.cols,CV_8UC3);
	Mat left(img_combine,Rect(0,0,img1.cols,img1.rows));
	image1.copyTo(left);
	Mat right(img_combine,Rect(img1.cols,0,img2.cols,img2.rows));
	image2.copyTo(right);
	namedWindow( "Display", WINDOW_AUTOSIZE );         // Create a window for display.
	imshow( "Display", img_combine );                   // Show our image inside it.
	waitKey(0);                                          // Wait for a keystroke in the window

	/*THIS PART IS JUST FOR FUN*/
	// -- Step 1: Detect the keypoints using STAR Detector
	const Mat img_1 =imread(filename1,1);
	const Mat img_2 =imread(filename2,1);
	vector<KeyPoint> keypoints_1,keypoints_2;
	Ptr<ORB> orb = ORB::create(40,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);
	// -- Stpe 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	orb->compute(img_1, keypoints_1, descriptors_1);
	orb->compute(img_2, keypoints_2, descriptors_2);
	//Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_HAMMING);
	std::vector<DMatch> mathces;
	matcher.match(descriptors_1, descriptors_2, mathces);
	Mat img_mathes;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, mathces, img_mathes);
	imshow("Mathces", img_mathes);
	waitKey(0);

	return 0;
}
