
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


class ImageProcess{
private:
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> image_sub;
    message_filters::Subscriber<sensor_msgs::Image> info_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync;
    int width;
    int height;
    cv_bridge::CvImagePtr color_ptr, depth_ptr;
    cv::Mat color_pic, depth_pic;
public:

    ImageProcess():nh("~"),
    image_sub(nh, "/camera/color/image_rect_color", 1),
    info_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1),
    sync(image_sub, info_sub, 10)
    {
        width = 640;
        height = 480;
        sync.registerCallback(boost::bind(&ImageProcess::callback, this,_1, _2));
        allocateMemory();
        resetParameters();
    }
    // 初始化各类参数以及分配内存
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        // labelComponents函数中用到了这个矩阵
        // 该矩阵用于求某个点的上下左右4个邻接点
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    // 初始化/重置各类参数内容
    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }
    void callback(const sensor_msgs::ImageConstPtr& color, const sensor_msgs::ImageConstPtr& depth)
    {

        color_ptr = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::BGR8);
        color_pic = color_ptr->image;
        depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
        depth_pic = depth_ptr->image;

        cv::Mat labelMat = cv::Mat(depth_pic.rows, depth_pic.cols, CV_32S, cv::Scalar::all(0));
        PointCloud::Ptr cloud ( new PointCloud );
        uint16_t* queueIndX = new uint16_t[depth_pic.rows*depth_pic.cols];
        uint16_t* queueIndY = new uint16_t[depth_pic.rows*depth_pic.cols];

        for (int m = 0; m < depth_pic.rows; m++){
            for (int n = 0; n < depth_pic.cols; n++){
                if (labelMat.at<int>(m, n) == 0)
                    labelComponents(m, n);
                // 获取深度图中(m,n)处的值
                if(depth_pic.ptr<float>(m)[n]>9000.)
                    depth_pic.ptr<float>(m)[n]=0.;//depth filter
                float d = depth_pic.ptr<float>(m)[n];//ushort d = depth_pic.ptr<ushort>(m)[n];
                // d 可能没有值，若如此，跳过此点
                if (d == 0.)
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
        pf.minSupport = 600;
        pf.windowWidth = 12;
        pf.windowHeight = 12;
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

    void labelComponents(int row, int col){
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;

        // 标准的BFS
        // BFS的作用是以(row，col)为中心向外面扩散，
        // 判断(row,col)是否是这个平面中一点
        while(queueSize > 0){
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // labelCount的初始值为1，后面会递增
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;

            // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
            // 遍历点[fromIndX,fromIndY]边上的四个邻点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){

                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;

                // 是个环状的图片，左右连通
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

                // 如果点[thisIndX,thisIndY]已经标记过
                // labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
                // 如果当前的邻点已经标记过，则跳过该点。
                // 如果labelMat已经标记为正整数，则已经聚类完成，不需要再次对该点聚类
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));

                // alpha代表角度分辨率，
                // X方向上角度分辨率是segmentAlphaX(rad)
                // Y方向上角度分辨率是segmentAlphaY(rad)
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                // 通过下面的公式计算这两点之间是否有平面特征
                // atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){
                    // segmentTheta=1.0472<==>60度
                    // 如果算出角度大于60度，则假设这是个平面
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }


        bool feasibleSegment = false;

        // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){
            // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;

            // 竖直方向上超过3个也将它标记为有效聚类
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;
        }

        if (feasibleSegment == true){
            ++labelCount;
        }else{
            for (size_t i = 0; i < allPushedIndSize; ++i){
                // 标记为-1的是需要舍弃的聚类的点，因为他们的数量小于30个
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = -1;
            }
        }
    }

};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_listener");
    ImageProcess IP;
    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

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
