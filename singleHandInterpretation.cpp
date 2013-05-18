/*
	Implementation of a hand gesture reconstruction algorithm
	by Pawel Borkowski 2012, borkowskip@gmail.com, paweldoes.com
*/
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include<vector>

#define PI 3.14;
using namespace cv;
using namespace std;
int tValue = 150;

struct skeleton{
	CvPoint centroid;
	vector<CvPoint> tips;
};

struct histogramPoint{
	CvPoint topPoint;
	int distance;

};

struct finger{
	CvPoint tip;
	int thickness;
};

struct histogram{
  //the value of all the top points - for drawing on the hand image
  CvPoint topPoints [360];
  //distances, both as an input for neural nets and for drawing the histogram
  int distances[360];	
  bool bin[360];
};


/*Returns a centroid from the given binary image (8, 1)*/
CvPoint getCentroid(IplImage* src){
	uchar *data;
	int totalX=0;
	int totalY=0;
	int pointsCount=0;
	data   = (uchar *)src->imageData;
	/*iterate through all of the points in src*/
	for(int y=0;y<480;y++){	 
		uchar* ptr=(uchar*)(src->imageData+y*src->widthStep);

       	for(int x=0; x<640;x++){
			int b=ptr[x];
			if(b!=0){
				pointsCount++;
				totalX+=x;
				totalY+=y;
			}
		}
	}
	int cX = 300;
	int cY = 300;

	if(pointsCount>0){
	   cX = (int)(totalX/pointsCount);
	   cY = (int)(totalY/pointsCount);
	 //  printf("CENTOIRD %d %d", cX, cY);
	}	 
	
	CvPoint centroid = cvPoint(cX, cY);
	return centroid;
}

void drawCountedFingers(IplImage* src, vector<finger> f){
	int no = f.size()-1;

    const char* text;
	if(no<=0)  text = "no fingers!";
	if(no==1)  text = "one finger";
	if(no==2)  text = "two fingers";
	if(no==3)  text = "three fingers";
	if(no==4)  text = "four fingers";
	if(no==5)  text = "full hand";
	if(no>5)   text = "too much noise";

	CvPoint pt = cvPoint(20, 450);
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 0.8, 0.2, 2, 8);
	cvPutText(src, text, pt, &font, Scalar(255, 255, 255));

}

/*count fingers works only on the binary histograms*/
vector<finger> countFingers(histogram h){
	int previous = -1;
        /*we have to remove possible noise - if the bar is smaller than 0.2 of the average remove it*/	
	int count=0;
	for(int d=0; d<360; d++);
		//if(h.bin) printf("0"); else printf("1");
	for(int d=0; d<360; d++){
		if(h.bin[d]==true) count++;
		//printf("c %d\n", count);
	}
	int average = (int)(count/5);
	//printf("avg %d\n", average);
 	
	vector<finger> fingers;
	int tempThickness =0; //calculate the thickness on the run
	bool current=false; //are we inside of one of the bars or somewhere else?
	for(int d=0; d<360; d++){
	   if(current){
	        if(!h.bin[d]){
		    if(tempThickness>(average*0.3)){
			CvPoint temp = h.topPoints[d];
		        finger f;
		        f.tip = temp;
		        fingers.push_back(f);
		    }
	            tempThickness=0;
		    
		    current = false;
		} else tempThickness++; 
	   } else {
	
             if(h.bin[d]){
	   	current=true;
		tempThickness++;
	     }  
	   }
	}		
	
	//printf("we've got %d fingers in here\n", fingers.size()-1);
	return fingers;
}
/*this is a mix of the function that creates skeletons and the one that creates histograms*/
histogram createSimplifiedHistogram(IplImage* src, IplImage* r, bool draw, int optimal){
	histogram h;
	IplImage* image = cvCreateImage(Size(500, 100), 32,1);
	if(draw){
		
	        cvNamedWindow("histogram", 1);
	}
	int smallest = 0; Size canvas = cvGetSize(src);	
	int height = canvas.height; int width  = canvas.width;
	int density=3;
       CvPoint centroid; centroid.x=50; centroid.y=50; centroid = getCentroid(src);
	/*finding the maximum R that we can do without going out of the image*/
	int maxR = 300;//Min(height-centroid.y, centroid.y, width-centroid.x, centroid.x);
	/*calculate the distances between the centroid circles*/
	int step = 5;//(int)(maxR/density);
	int currentStep = optimal;
	int current = 0;
	int lastAngle =0;
	CvPoint lastPoint;
		//  printf("--------------------\n");
		for(int d=0; d<360; d++){
			h.bin[d]=false;
			/*calculate the point*/
			float angle = d*PI;
	
			float hD = sin(angle/180);
			float cosD = cos(angle/180);

			int x = (int)(cosD*currentStep)+centroid.x;
			int y = (int)(sinD*currentStep)+centroid.y;
			//printf("angle = %d / %f, (%d, %d)\n", d, sinD, x, y);
			CvPoint currentPoint;
			currentPoint.x = x; currentPoint.y = y;
			
			/*check if the point is a part of the hand*/
			uchar* ptr=(uchar*)(src->imageData+y*src->widthStep);
	 		int b=ptr[x]; 
			
		         //if(b>0)   printf("0\n", b); else  printf("1\n", b);
	 	        if(current==0){
	     		  if(b>0){
				 lastAngle = d;
              			 lastPoint.x = x;
	      		 	 lastPoint.y = y;
              		 	 current=1;
				
	    		  }
	   		 } else {
			     if (b == 0) {
			         int lastX = lastPoint.x;
			         int lastY = lastPoint.y;
			         int cSY = lastX + x;
			         int cSX = lastY + y;
			         int centreX = (int)(cSX / 2);
			         int centreY = (int)(cSY / 2);

			         CvPoint to = cvPoint(x, y);
			         CvPoint srodek = cvPoint(centreX, centreY);
			         cvLine(r, lastPoint, to, cvScalar(255, 255, 255, 0), 2, 8, 0);
			         cvCircle(r, to, 1, cvScalar(255, 255, 255, 0), 10, 10, 0);
			         current = 0;


			         //printf("from %d to %d", lastAngle, d);
			         if (draw) {
			             CvPoint a = cvPoint(lastAngle, 100);
			             CvPoint a2 = cvPoint(lastAngle, 0);
			             CvPoint b2 = cvPoint(d, 0);
			             CvPoint b = cvPoint(d, 100);
			             CvPoint pts[] = {
			                 a, a2, b2, b
			             };
			             cvFillConvexPoly(image, pts, 4, cvScalar(255));
			         }
			         for (int i = lastAngle; i < d; i++) {
			             int lastX = lastPoint.x;
			             int lastY = lastPoint.y;

			             int sumX = lastX + x;
			             int sumY = lastY + y;

			             int centreX = (int)(sumX / 2);
			             int centreY = (int)(sumY / 2);
			             h.distances[i] = 100;
			             h.topPoints[i] = cvPoint(sumX, sumY);
			             h.bin[i] = true;
			         }
			     }

			 }
	
	}
	if(draw) cvShowImage("histogram", image);
	return h;


}
histogram createHistogram(IplImage* src, IplImage* r){//, CvPoint centroid, int density){ //vector<vector<histogramPoint>> 
	histogram h;
	
	int smallest = 0;
 	Size canvas = cvGetSize(src);	
	int height = canvas.height;
	int width  = canvas.width;
	int density=3;
	CvPoint centroid; centroid = getCentroid(src);
	/*finding the maximum R that we can do without going out of the image*/
	int maxR = 300;//Min(height-centroid.y, centroid.y, width-centroid.x, centroid.x);
	/*calculate the distances between the centroid circles*/
	int step = 5;//(int)(maxR/density);
	int currentStep = 150;
	
	for(int i=0; i<360; i++){
		h.distances[i]=0;
		
	}

	while(currentStep<maxR){
		vector<histogramPoint> tempLayer;

		
		for (int d = 0; d < 360; d++) {
	    /*calculate the point*/
		    float angle = d * PI;

		    float sinD = sin(angle / 180);
		    float cosD = cos(angle / 180);

		    int x = (int)(cosD * currentStep) + centroid.x;
		    int y = (int)(sinD * currentStep) + centroid.y;
		    //printf("angle = %d / %f, (%d, %d)\n", d, sinD, x, y);
		    CvPoint currentPoint;
		    currentPoint.x = x;
		    currentPoint.y = y;

		    /*check if the point is a part of the hand*/
		    uchar * ptr = (uchar * )(src - > imageData + y * src - > widthStep);
		    int b = ptr[x];
		    //if(b!=255&&b!=0) ;//printf("%d\n", b);
		    if (b > 0) {
		        CvPoint p;

		        p.x = x;
		        p.y = y;
		        h.topPoints[d] = p;
		        h.distances[d] = currentStep;
		        //cvCircle(r, currentPoint, 1, cvScalar(0, 0, 255, 0), 1, 1, 0);   

		    }
	}
	currentStep += step;
	}
	return h;
}
/*Function draws the parameter histogram and opens an additional window to show it*/
void drawHistogram(histogram h){
	IplImage* image;
	image = cvCreateImage(Size(500, 100), 8,1);
	for(int i=0; i<360; i++){
		CvPoint from; from.x=i; from.y=100;
		CvPoint to;   to.x=i;   to.y=h.distances[i];
		CvPoint pts[] = {from, to};
		cvFillConvexPoly(image, pts, 2, cvScalar(255));
		//cvLine(image, from, to,	cvScalar(255, 255, 255, 0), 1, 8, 0);
	}
	int key = waitKey(40);	
	
	cvShowImage("histogram", image);
	cvReleaseImage(&image);
}

skeleton reconstructSkeleton(IplImage* src){
	skeleton result;
	/* 1. calculate the centre of gravity */

	CvPoint c = getCentroid(src);
	result.centroid = getCentroid(src);
	
	int cX = result.centroid.x;
	int cY = result.centroid.y;
  /*2. circular loop*/
   int current = 0;
   CvPoint lastPoint = cvPoint(0, 0);
	 

     for(int i=1; i<2; i++){
	int xT, yT;
	int length = 150;
	float angle = 0.0;
	float angle_stepsize = 0.01;
	while (angle < 2 * 3.14) {

	    xT = (length + i * 20) * cos(angle) + cX;
	    yT = (length + i * 20) * sin(angle) + cY;
	    CvPoint tPoint = cvPoint(xT, yT);


	    uchar * ptr = (uchar * )(src - > imageData + yT * src - > widthStep);
	    int b = ptr[xT];


	    if (current == 0) {
	        if (b != 0) {
	            lastPoint.x = xT;
	            lastPoint.y = yT;
	            current = 1;
	        }

	    } else {
	        if (b == 0) {
	            int lastX = lastPoint.x;
	            int lastY = lastPoint.y;

	            int sumX = lastX + xT;
	            int sumY = lastY + yT;

	            int centreX = (int)(sumX / 2);
	            int centreY = (int)(sumY / 2);


	            CvPoint cPoint = cvPoint(centreX, centreY);
	            result.tips.push_back(cPoint);
	            current = 0;
	        }
	    }
	    angle += angle_stepsize;
	}
	}
	return result;
	}
void drawSkeleton(skeleton input, IplImage* src){
	
	cvCircle(src, input.centroid, 10, cvScalar(0, 0, 255, 0), 2, 5, 0);   
	
	for(int i=0; i<input.tips.size(); i++){
		cvCircle(src, input.tips[i], 10, cvScalar(0, 0, 255, 0), 2, 5, 0);   
		cvLine(src, input.tips[i], input.centroid, cvScalar(255, 255, 255, 0), 1, 8, 0); 
	}

}

void cleanImage(IplImage* input, IplImage* dst){

	IplImage* r = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	IplImage* g = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	IplImage* b = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);

	cvSplit(input, r, g, b, NULL); //split the channels
	IplImage* temp = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvAddWeighted(r, 1./3., g, 1./3., 0.0, temp);
	cvAddWeighted(temp, 2./3., b, 1./3., 0.0, temp);

	cvThreshold(temp, dst, 100, 255, CV_THRESH_BINARY);
	cvReleaseImage(&r);
	cvReleaseImage(&g);
	cvReleaseImage(&b);
	cvReleaseImage(&temp);
}


int main()
{

    IplImage* frame;
    IplImage* tresh;
    CvCapture* capture = cvCreateCameraCapture(1);
    cvNamedWindow("image",1);	
    cvNamedWindow("tresh",1);	
   
    int i=0;
    /*until the aplication is terminated, proceed with the calculation*/
    while(1){
	
        frame = cvQueryFrame(capture); 
		tresh = cvCreateImage(cvGetSize(frame), 8,1);
		cleanImage(frame, tresh);
		//skeleton temp = reconstructSkeleton(tresh);
		//drawSkeleton(temp, frame);
		histogram h2 = createHistogram(tresh, frame);
		int optimalStep=0;
		int stepSum=0;
		for(int i=0; i<360; i++){
			stepSum+=h2.distances[i];
			//printf("DISTANCE : %d", h2.distances[i]);
		}
		optimalStep =  (int)((stepSum/360)*0.5)+110;
		//printf("OPTIMAL : %d", optimalStep);
		histogram h =createSimplifiedHistogram(tresh, frame, 1, optimalStep);
		vector<finger> f= countFingers(h);
		drawCountedFingers(frame, f);
		if(i==10){        
		// drawHistogram(h);
		 i=0;
		}else i++;
		cvShowImage("image", frame);
		cvShowImage("tresh", tresh);
	        if(waitKey(30) >= 'q') break;
    }
    cvReleaseImage(&tresh);
    cvReleaseImage(&frame);
    cvReleaseCapture(&capture);
    return(0);
}
