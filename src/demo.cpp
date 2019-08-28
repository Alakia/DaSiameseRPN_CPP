//
// Created by marco on 19-8-13.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <random>
#include <fstream>
#include "stdio.h"

#include "Storage.h"
#include "SiamTracker.h"
#include "SiameseRPN.h"

using namespace std;
using namespace cv;

bool ldown = false;
bool lup = false;
bool getroi = false;
Point corner1init;
Point corner2init;
Mat imginit;
Mat frame, framecol;
Rect2f box;
int framecnt = 0;

static void mouse_callback(int event, int x, int y, int, void *)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        ldown = true;
        corner1init.x = x;
        corner1init.y = y;
        cout << "Corner 1 recorded at " << corner1init << endl;
    }
    if(event == EVENT_LBUTTONUP)
    {
        if(abs(x - corner1init.x) > 5 && abs(y - corner1init.y) > 5)
        {
            lup = true;
            corner2init.x = x;
            corner2init.y = y;
            cout << "Corner 2 recorded at " << corner2init << endl << endl;
        }
        else
        {
            cout << "Please select a bigger region" << endl;
            ldown = false;
        }
    }
    if(ldown == true && lup == false)
    {
        Point pt;
        pt.x = x;
        pt.y = y;
        Mat local_img = imginit.clone();
        rectangle(local_img, corner1init, pt, Scalar(0, 0, 255));
        imshow("KCF", local_img);
    }
    if(ldown == true && lup == true)
    {
        box.width = abs(corner1init.x - corner2init.x);
        box.height = abs(corner1init.y - corner2init.y);
        box.x = min(corner1init.x, corner2init.x);
        box.y = min(corner1init.y, corner2init.y);
        ldown = false;
        lup = false;
        getroi = true;
    }
}

int main()
{
    VideoCapture cap;
    cap.open("/home/marco/Videos/track/test3.mp4");

    if(!cap.isOpened()) {
        cap.open(0);
    }

    cv::Mat imgcolor;
    cap >> imgcolor;
    int w = 1280, h = 960;
    namedWindow("KCF");
    setMouseCallback("KCF", mouse_callback);

    while(1) {
        if(getroi){
            cv::destroyWindow("KCF");
            break;
        }
        cap >> imginit;
        imshow("KCF", imginit);
        waitKey(0);
    }

    IDataStorage::Ptr datastorage = std::make_shared<IDataStorage>(IDataStorage());
    Matimage::Ptr pmat = std::make_shared<Matimage>(Matimage(imgcolor));
    datastorage->Set("init_image", pmat);
    InputInfo info;
    info.width_ = w;
    info.height_ = h;
    info.conf_ = -1;
    info.bbox_ = box;
    InputInfo::Ptr pinfo = std::make_shared<InputInfo>(info);
    datastorage->Set("info", pinfo);

    SiameseRPNProcessor siamtracker;
    siamtracker.Init(datastorage);

    cout << "Start the tracking process.\n" << endl;
    double tsum = 0.0;

    int fcnt = 0;
    for ( ; ; ) {
        cap >> frame;
        framecnt++;
        printf("current frame %d\n", framecnt);

        if (frame.rows == 0 || frame.cols == 0){
            break;
        }

        double t = (double)cv::getTickCount();
        pinfo->conf_ = -1;
        Matimage::Ptr pframe = std::make_shared<Matimage>(Matimage(frame));
        datastorage->Set("frame", pframe);
        datastorage->Set("info", pinfo);
        siamtracker.Process(datastorage);
        t = ((double)getTickCount() - t) * 1000 / getTickFrequency();
        tsum += t;

        if(!pinfo->flag_){
            std::cout << "tracker lost, reinit." << std::endl;
            putText(frame, "Lost", Point2i(100, 100), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2);
        }
        else {
            cv::Rect2f bbox = pinfo->bbox_;
            rectangle(frame, bbox, Scalar(0, 255, 255), 2, 1);
            rectangle(framecol, bbox, Scalar(0, 255, 255), 2, 1);
            putText(frame, "Tracking", Point2i(100, 100), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2);
        }

        imshow("KCF",frame);
        fcnt++;
        char k = waitKey(10);
        if(k == 27)  break;
    }
    printf("time avg = %f\n", tsum / framecnt);
    return 0;
}
