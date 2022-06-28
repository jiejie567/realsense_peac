
#include <time.h>
#include <iostream>
 
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>

#include "AHCPlaneFitter.hpp"

using namespace  cv;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
pcl::visualization::CloudViewer viewer("Cloud Viewer: Rabbit");     //创建viewer对象
// 相机内参
const double camera_factor = 1000;
const double camera_cx = 333.891;
const double camera_cy = 254.687;
const double camera_fx = 597.53;
const double camera_fy = 597.795;
const float max_use_range = 10;
 
void callback(const sensor_msgs::ImageConstPtr& color, const sensor_msgs::ImageConstPtr& depth)
{
    //ROS_INFO("RECEIVE Image...");
//    try
//    {
//        cv::imshow("view", cv_bridge::toCvShare(color, "bgr8")->image);
//    }
//    catch (cv_bridge::Exception& e)
//    {
//        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color->encoding.c_str());
//    }
//    try
//    {
//        //cv::imshow("depthview", cv_bridge::toCvShare(depth, "16UC1")->image);
//        cv::imshow("depthview", cv_bridge::toCvShare(depth, sensor_msgs::image_encodings::TYPE_32FC1)->image);
//    }
//    catch (cv_bridge::Exception& e)
//    {
//        ROS_ERROR("Could not convert from '%s' to '16UC1'.", depth->encoding.c_str());
//    }
 
    //ointCloud
    cv_bridge::CvImagePtr color_ptr, depth_ptr;
    cv::Mat color_pic, depth_pic;
    color_ptr = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::BGR8);
    color_pic = color_ptr->image;
    depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_pic = depth_ptr->image;
    //cout<<"channels"<<depth_pic.depth()<<endl;
 
    PointCloud::Ptr cloud ( new PointCloud );
 
    for (int m = 0; m < depth_pic.rows; m++){
        for (int n = 0; n < depth_pic.cols; n++){
            // 获取深度图中(m,n)处的值
            if(depth_pic.ptr<float>(m)[n]>4096)
                depth_pic.ptr<float>(m)[n]=0;//depth filter
            float d = depth_pic.ptr<float>(m)[n];//ushort d = depth_pic.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            pcl::PointXYZRGB p;
 
            // 计算这个点的空间坐标
            p.z = -double(d) / camera_factor;
//            if(m==(int)(depth_pic.rows/2)&&n==(int)(depth_pic.cols/2))
//                cout<<double(d)<<endl;
            p.x = -(n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
 
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = color_pic.ptr<uchar>(m)[n*3];
            p.g = color_pic.ptr<uchar>(m)[n*3+1];
            p.r = color_pic.ptr<uchar>(m)[n*3+2];
 
            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    }

    // ahc 
    struct OrganizedImage3D {
    const cv::Mat_<cv::Vec3f>& cloud_peac;
    //note: ahc::PlaneFitter assumes mm as unit!!!
    OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c): cloud_peac(c) {}
    inline int width() const { return cloud_peac.cols; }
    inline int height() const { return cloud_peac.rows; }
    inline bool get(const int row, const int col, double& x, double& y, double& z) const {
        const cv::Vec3f& p = cloud_peac.at<cv::Vec3f>(row,col);
        x = p[0];
        y = p[1];
        z = p[2];
        return z > 0 && isnan(z)==0; //return false if current depth is NaN
    }
    };  
    typedef ahc::PlaneFitter< OrganizedImage3D > PlaneFitter;

    cv::Mat_<cv::Vec3f> cloud_peac(depth_pic.rows, depth_pic.cols);
    for(int r=0; r<depth_pic.rows; r++)
    {
        const float* depth_ptr = depth_pic.ptr<float>(r);
        cv::Vec3f* pt_ptr = cloud_peac.ptr<cv::Vec3f>(r);
        for(int c=0; c<depth_pic.cols; c++)
        {
            float z = (float)depth_ptr[c]/camera_factor;
            if(z>max_use_range){z=0;}
            pt_ptr[c][0] = (c-camera_cx)/camera_fx*z*1000.0;//m->mm
            pt_ptr[c][1] = (r-camera_cy)/camera_fy*z*1000.0;//m->mm
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }
    PlaneFitter pf;
    pf.minSupport = 3000;
    pf.windowWidth = 20;
    pf.windowHeight = 20;
    pf.doRefine = true;

    cv::Mat seg(depth_pic.rows, depth_pic.cols, CV_8UC3);
    OrganizedImage3D Ixyz(cloud_peac);
    pf.run(&Ixyz, 0, &seg);
    cv::imshow("view",seg);
 
    // cv::imshow("view", color_pic);
    cv::imshow("depthview", depth_pic/4096.);
    viewer.showCloud(cloud);
 
    char c = (char)cv::waitKey(50);//得到键值
    static bool startsave=false;
    if (c == 'h')
    {
        startsave = true;
        ROS_INFO("Start write Image...");
    }
    if(c=='f')
    {
        startsave= false;
        ROS_INFO("Stop write Image...");
    }
    if(startsave||c == 'a')
    {
        time_t t = time(NULL);
        struct tm* stime=localtime(&t);
        char tmp[32]{0};
        snprintf(tmp,sizeof(tmp),"%04d%02d%02d%02d%02d%02d",1900+stime->tm_year,1+stime->tm_mon,stime->tm_mday, stime->tm_hour,stime->tm_min,stime->tm_sec);
        cout<<tmp<<endl;
//        cout<<depth_pic.channels()<<endl;
//        cout<<depth_pic.depth()<<endl;
        cout<<"d center point"<<depth_pic.at<float>(240,320)<<endl;
        std::string croad="/home/project/myimage/all/color+"+std::string(tmp)+".png";
        std::string droad="/home/project/myimage/all/depth+"+std::string(tmp)+".png";
        std::string droadtif="/home/cq/project/myimage/all/depth+"+std::string(tmp)+".tif";
//        std::string croad="/home/project/myimage/color1.png";
//        std::string droad="/home/project/myimage/depth1.png";
//        std::string droadtif="/home/project/myimage/depth1.tif";
        cv::imwrite(croad,color_pic);//
 
        //********************************
        Mat dep8u(depth_pic.rows,depth_pic.cols,CV_8UC4,depth_pic.data);
        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        Mat depth832(depth_pic.rows,depth_pic.cols,CV_32FC1,dep8u.data);
        cout<<"depth832 center point"<<depth832.at<float>(240,320)<<endl;
 
        //***save depth
 
        cv::imwrite(droad,dep8u,compression_params);
        cv::imwrite(droadtif,depth_pic);

        //evluate ***********
//        cv::Mat depth1 = imread("/home/cq/project/myimage/depth1.png",IMREAD_UNCHANGED);
//        Mat depth32(depth_pic.rows,depth_pic.cols,CV_32FC1,depth1.data);
//
//        cout<<depth32.channels()<<endl;
//        cout<<depth32.depth()<<endl;
//        cout<<"d center point"<<depth32.at<float>(240,320)<<endl;
//
//        cv::Mat depthtif = imread("/home/cq/project/myimage/depth1.tif",IMREAD_UNCHANGED);
//        cout<<"tif center point"<<depthtif.at<float>(240,320)<<endl;
        ROS_INFO("write Image...");
    }
    color_pic.release();
    depth_pic.release();
 
}
 
int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    cv::namedWindow("view");
    cv::startWindowThread();
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_rect_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> info_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    ros::spin();
    cv::destroyWindow("view");
 
}
