#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include "matching.h"

using namespace std;
using namespace cv;


int main()
{
	vector< Mat > origin_img, moved_img;
	
	if ( !GetSourceImage(origin_img, moved_img) )
	{
		cerr << __func__ << " : image load fail" <<endl;
		return -1;
	}

	for (int i = 0; i < static_cast<int>(origin_img.size()) ; ++i)
	{
		Mat result_image;
		if (!FaceDetect(origin_img[i]) || !FaceDetect(moved_img[i]))
		{
			cerr << __func__ << " : Not Detect Face" << endl;
			continue;
		}
		ResizeImage(origin_img[i], moved_img[i]);

		int match_point = matching(origin_img[i], moved_img[i], result_image);
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