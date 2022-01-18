#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "matching.h"
#include <thread>

using namespace std;
using namespace cv;

int main()
{
	vector< Mat > origin_img, moved_img;

	if (!GetSourceImage(origin_img, moved_img))
	{
		cerr << __func__ << " : image load fail" << endl;
		return -1;
	}

	for (int i = 0; i < static_cast<int>(origin_img.size()); ++i)
	{
		Mat result_image;
		vector<Rect> face_detect, target_face;
		target_face = FaceDetect(origin_img[i]);
		face_detect = FaceDetect(moved_img[i]);
		int dectect_size = static_cast<int>(face_detect.size());
		if (dectect_size == 0)
		{
			std::cerr << __func__ << " : Not find face" << std::endl;
			continue;
		}

		int match_point = -1;
		int idx_match_point = -1;

		for (int j = 0; j < dectect_size; ++j)
		{

			Mat crop_origin_img, crop_moved_img, sub_result_image;

			if (!ResizeImage(moved_img[i], origin_img[i], face_detect[j], target_face[0], crop_origin_img, crop_moved_img))
			{
				continue;
			}
			int sub_match_point = matching(crop_moved_img, crop_origin_img, sub_result_image);
			if (match_point < sub_match_point)
			{
				match_point = sub_match_point;
				result_image = sub_result_image;
			}

		}


		if (match_point == 0)
		{
			cout << __func__ << "_ no match_point " << endl;
			continue;
		}
		cout << "match_point : " << match_point << endl;

		String match_text = "match_point : " + to_string(match_point);

		putText(result_image, match_text, Point(10, 20), 1, 1.2f, Scalar::all(255));
		namedWindow("match");
		imshow("match", result_image);
		imwrite("matching_images_" + to_string(i) + ".jpg", result_image);
		waitKey(0);
		destroyAllWindows();
	}

	return 0;
}