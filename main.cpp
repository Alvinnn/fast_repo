
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
    string filename2="/Users/macbookpro/Downloads/lena.jpg";
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
    //cout<<"haha: "<<keypoint.at(1).pt<<endl;
//	uchar* d_data;                // create a pointer
//	size_t imSize=image.cols*image.rows;
//	Corner* h_corner=new Corner[imSize];
//	Corner* d_corner;
//	checkCudaErrors(cudaMalloc((void**) &d_corner,sizeof(Corner)*imSize));
//	checkCudaErrors(cudaMalloc((void**) &d_data, sizeof(uchar)*imSize)); // create memory on the gpu and pass a pointer to the host
//	checkCudaErrors(cudaMemcpy(d_data, image.data, sizeof(uchar)*imSize, cudaMemcpyHostToDevice));// copy from the image data to the gpu memory you reserved
//	dim3 blocksize(16,16);
//	dim3 gridsize((image.cols-1)/blocksize.x+1, (image.rows-1)/blocksize.y+1, 1);
//	fast<<<gridsize,blocksize>>>(d_data, image.cols, image.rows,d_corner,gridsize.x,gridsize.y,threshold); // processed data on the gpu
//	//checkCudaErrors(cudaDeviceSynchronize());
//	cudaEventRecord(stop);	cudaEventSynchronize(stop);
//	nms<<<gridsize,blocksize>>>(d_data,d_corner,image.cols,image.rows);
//	checkCudaErrors(cudaMemcpy(h_corner,d_corner,sizeof(Corner)*imSize,cudaMemcpyDeviceToHost));
//	float elptime;
//	cudaEventElapsedTime(&elptime,start,stop);
//	//show the corner in the image
//	Mat img_color = imread(filename,1);
//	Mat img_rotate= imread("/Users/macbookpro/Downloads/lena.jpg",1);
//	Size size1=img_color.size();
//	Size size2=img_rotate.size();
//
//	int point=0;
//	for(int i=0;i<imSize;i++)
//	{
//		if(h_corner[i].set!=0)
//		{
//			int x=i%image.cols;
//			int y=i/image.cols;
//			circle(img_color,Point(x,y),1,Scalar(0,255,0),-1,8,0);
//			point++;
//		}
//	}
//
//	for(int i=0;i<imSize;i++)
//	{
//		if(h_corner[i].set!=0)
//		{
//			int m=i%image.cols;
//			int n=i/image.cols;
//			circle(img_rotate,Point(m,n),1,Scalar(0,255,0),-1,8,0);
//
//		}
//	}
//
//	Mat img_combine(img_color.rows,img_color.cols+img_rotate.cols,CV_8UC3);
//	Mat left(img_combine,Rect(0,0,img_color.cols,img_color.rows));
//	img_color.copyTo(left);
//	Mat right(img_combine,Rect(img_color.cols,0,img_rotate.cols,img_rotate.rows));
//	img_rotate.copyTo(right);
//
//	cout<<"points:"<<point<<endl;
//	cout<<"Elapsed time:"<<elptime<<"ms"<<endl;
//	//printf("%x\n",0x7|((10>1)<<3));
//	//cout<<"the size of: "<<sizeof(corner)<<endl;

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
	namedWindow( "Display", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display", img_combine );                   // Show our image inside it.
	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}
