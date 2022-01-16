#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include <list>
#include <vector>
#include <iostream>
#include <Windows.h>
#include <io.h>

#define IMAGE_DIRECTORY "original_images/"
#define MOVED_IMAGE_DIRECTORY "moved_images/"
#define IMAGE_EXTENSION "*.jpg"
#define HAARSCASCADE_XML "data/haarcascades/haarcascade_frontalface_alt.xml"
#define KEYPOINT_COUNT 10000
#define INLIER 20.f
#define RATIO 0.8f 

std::vector<std::string> get_files_directory(const std::string& file_path, const std::string& include_word)
{
    std::string searching = file_path + include_word;
    std::vector<std::string> file_names;


    _finddata_t find_data;

    intptr_t handle = _findfirst(searching.c_str(), &find_data);
    if (handle == -1)
    {
        std::cout << __func__ << " : No "<< file_path <<" files " << std::endl;
        return file_names;
    }

    int check_file_count = 0;
    while (check_file_count != -1)
    {
        file_names.push_back(find_data.name);
        check_file_count = _findnext(handle, &find_data);
    }

    _findclose(handle);
    return file_names;
}




bool GetSourceImage(std::vector< cv::Mat >& origin_image,std::vector< cv::Mat >& moved_image)
{

    std::vector<std::string> origin_image_names = get_files_directory( IMAGE_DIRECTORY, IMAGE_EXTENSION);
    std::vector<std::string> moved_image_names = get_files_directory( MOVED_IMAGE_DIRECTORY, IMAGE_EXTENSION);
    if (origin_image_names.size() == 0 || moved_image_names.size() == 0)
    {
        std::cerr << __func__ << " : Not load image" << std::endl;
        return false;
    }
    int name_count = static_cast<int>( origin_image_names.size() );

    while ( name_count != 0)
    {
        origin_image.push_back( cv::imread(IMAGE_DIRECTORY + origin_image_names[name_count - 1] ) );
        moved_image.push_back( cv::imread(MOVED_IMAGE_DIRECTORY + moved_image_names[name_count - 1] ) );
        name_count --;
    }

	return true;
}

int FaceDetect(cv::Mat& image)
{
    std::vector<cv::Rect> faces;
    cv::CascadeClassifier face_cascade;
    
    face_cascade.load(cv::samples::findFile(HAARSCASCADE_XML));
    if (face_cascade.empty())
    {
        std::cerr << "Error Loading XML file" << std::endl;
        return 0;
    }
    face_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
    int face_count = static_cast<int>(faces.size());
    if (face_count == 0)
    {
        std::cerr << __func__ << " : Not find face" << std::endl;
        return 0;
    }
    image = image(cv::Rect(cv::Point(faces[0].x, faces[0].y), cv::Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height)));

    return 1;

}

void ResizeImage(cv::Mat& origin_img, cv::Mat& moved_img)
{

    int width = origin_img.cols > moved_img.cols ? origin_img.cols : moved_img.cols;
    int height = origin_img.rows > moved_img.rows ? origin_img.rows : moved_img.rows;
    if (width < 20 || height < 20)
    {
        std::cerr << __func__ << " : image size too small (" << width << ", " << height << ")" << std::endl;
    
    }
    cv::resize(origin_img, origin_img, cv::Size(width, height));
    cv::resize(moved_img, moved_img, cv::Size(width, height));
}

int RatioTest(std::vector< std::vector<cv::DMatch> >& matches, std::vector<cv::KeyPoint>& matched1, std::vector<cv::KeyPoint>& matched2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2)
{
    int keymatch_size = static_cast<int>(matches.size());
    std::cout << "keymatch_size : " << keymatch_size << std::endl;
    if (keymatch_size == 0) return 0;

    for (size_t i = 0; i < matches.size(); i++) {
        cv::DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;
        if (dist1 < RATIO * dist2) {
            matched1.push_back(keypoints1[first.queryIdx]);
            matched2.push_back(keypoints2[first.trainIdx]);
        }
    }
    return 1;
}

void DistanceTest(std::vector< cv::KeyPoint >& matched_origin, std::vector< cv::KeyPoint >& matched_moved, std::vector< cv::KeyPoint >& inliers1, std::vector< cv::KeyPoint >& inliers2, std::vector< cv::DMatch >& good_matching)
{
    for (size_t i = 0; i < matched_origin.size(); i++) {
        cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched_origin[i].pt.x;
        col.at<double>(1) = matched_origin[i].pt.y;
        double dist = sqrt(pow(col.at<double>(0) - matched_moved[i].pt.x, 2) + pow(col.at<double>(1) - matched_moved[i].pt.y, 2));
        if (dist < INLIER) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched_origin[i]);
            inliers2.push_back(matched_moved[i]);
            good_matching.push_back(cv::DMatch(new_i, new_i, 0));
        }
    }
}


int matching(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& result_image)
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::Feature2D> feature = cv::SIFT::create(KEYPOINT_COUNT);

    feature->detectAndCompute(image1, cv::Mat(), keypoints1, desc1);
    feature->detectAndCompute(image2, cv::Mat(), keypoints2, desc2);
    cv::BFMatcher matcher(cv::NORM_L2);

    std::vector< std::vector<cv::DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    std::vector<cv::KeyPoint> matched1, matched2;
    if (!RatioTest(nn_matches, matched1, matched2, keypoints1, keypoints2))
    {
        std::cerr << __func__ << "no match keypoint" << std::endl;
        return 0;
    }

    std::vector<cv::DMatch> good_matches;
    std::vector<cv::KeyPoint> inliers1, inliers2;
    DistanceTest(matched1, matched2, inliers1, inliers2, good_matches);

    cv::drawMatches(image1, inliers1, image2, inliers2, good_matches, result_image);
    return static_cast<int>(inliers1.size());
}