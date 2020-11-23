#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <dirent.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/math/distributions/normal.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/flann/flann.hpp"

#include "FastEMD/emd_hat.hpp"
#include "FastEMD/emd_hat_signatures_interface.hpp"
#include "RubnerEMD/emd.hpp"
#include "FastEMD/tictoc.hpp"

using namespace cv;
using namespace std;

namespace fs = boost::filesystem;

#define IMAGE_WIDTH 1000

// TODO: config
/* CONFIGURATIONS */
int keypoint_filter_threshold = 1;
bool filter_kp = 0;
int sort_type = -1;
bool show_images = 0;
/* END CONFIGURATIONS */

void help();
void find_images(const char *path);
void print_progress(time_t start, int total, int& completed, int per_cout);
void print_keypoints(const vector<KeyPoint>& kp);

bool extract_descriptors(int index, int fc, vector<KeyPoint>& kps, Mat& descs);
void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int fc);
void filter_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& f);
void calc_response_density_img(const Mat& img, vector<KeyPoint> kp, Mat& ds);
void calc_reponse_response_density_space(const vector<KeyPoint>& kp);
void calc_scale_density_img(const Mat& img, vector<KeyPoint> kp, Mat& ds);
void calc_scale_response_density_space(const vector<KeyPoint>& kp);
float keypoint_dist(const KeyPoint& k1, const KeyPoint& k2);
float point_dist(const Point2f& p1, const Point2f& p2);
bool response_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_density(const KeyPoint& k);
bool density_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_scale_density(const KeyPoint& k);
bool scale_density_comparator(const KeyPoint& p1, const KeyPoint& p2);
bool scale_comparator(const KeyPoint& p1, const KeyPoint& p2);

vector<fs::path> img_paths;

float response_density_space[IMAGE_WIDTH][IMAGE_WIDTH];
float scale_response_density_space[IMAGE_WIDTH][IMAGE_WIDTH];
Mat current_img;

vector<KeyPoint> filtered_keypoints;
struct response_density_classcomp {
	bool operator()(const int& lhs, const int& rhs) const {
		KeyPoint kp1, kp2;
		kp1 = filtered_keypoints[lhs];
		kp2 = filtered_keypoints[rhs];
		return density_comparator(kp1, kp2);
	}
};
struct scale_response_density_classcomp {
	bool operator()(const int& lhs, const int& rhs) const {
		KeyPoint kp1, kp2;
		kp1 = filtered_keypoints[lhs];
		kp2 = filtered_keypoints[rhs];
		return scale_density_comparator(kp1, kp2);
	}
};

int main(int argc, char *argv[]) {
	if (argc < 4) {
		help();
		return -1;
	}

	int fc = atoi(argv[1]);
	sort_type = atoi(argv[2]);
	const char *imgs_path = argv[3];
	const char *index_path = argv[4];

	cout << "Reading image paths from " << imgs_path << endl;
	string img_path;
	int img_count;
	ifstream imgs_file(imgs_path);
	imgs_file >> img_count;
	for (int i = 0; i < img_count; ++i) {
		imgs_file >> img_path;
		img_paths.push_back(img_path);
	}
	cout << img_count << " images found" << endl;

	int total_process = img_count;
	int completed_process = 0;
	time_t start = time(0);

	cout << "Building index..." << endl;

	tictoc timer;
	timer.tic();

	vector<KeyPoint> kps;
	Mat descs;
	FileStorage fs(index_path, FileStorage::WRITE);
	fs << "index" << "[";
	for (int i = 0; i < img_count; ++i) {
		if (extract_descriptors(i, fc, kps, descs)) {
			fs << "{";
			fs << "path" << img_paths[i].string();
			fs << "keypoints" << kps;
			fs << "descriptors" << descs;
			fs << "}";
		}
	}
	fs << "]";
	fs.release();

	timer.toc();
	cout << "Time in seconds: " << timer.totalTimeSec() << endl;

	return 0;
}

void help() {
	cout << "Usage: ./build_index <fc> <sort_type> <imgs_path> <index_path>" << endl;
}

void print_progress(time_t start, int total, int& completed, int per_count) {
	++completed;
	if (completed % per_count != 0)
		return;
	int remaining_process = total - completed;
	time_t current = time(0);
	float elapsed_time = difftime(current, start);
	float remaining_time = elapsed_time * remaining_process / completed;
	cout << fixed << setprecision(3);
	cout << "\rRemaining time: ";
	cout << remaining_time / 60 << "min";
}

void print_keypoints(const vector<KeyPoint>& kp) {
	KeyPoint k;
	cout << setprecision(3) << fixed;
	for (int i = 0; i < (int) kp.size(); ++i) {
		k = kp[i];
		cout << "(" << k.pt.x << "\t" << k.pt.y << "):\t" << k.response << "\t" << k.size << "\t" << get_scale_density(k) << endl;
	}
}

void find_images(const char *path) {
	fs::path dir_path(path);
	fs::recursive_directory_iterator end_iter;

	for (fs::recursive_directory_iterator iter(dir_path); iter != end_iter; ++iter) {
		if (fs::is_regular_file(iter->status())) {
			if (!boost::algorithm::ends_with(iter->path().c_str(), ".png"))
				continue;
			img_paths.push_back(iter->path());
		}
	}

	sort(img_paths.begin(), img_paths.end());
}

bool extract_descriptors(int index, int fc, vector<KeyPoint>& kps, Mat& descs) {
	string img_path = img_paths[index].string();
	fs::path yaml = img_paths[index];
	string yaml_path = yaml.replace_extension(".yml").string();
	
	// cout << "Calc img matrix: " << img_path << endl;
	current_img = imread(img_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (!current_img.data) {
		cout << "Error reading image: " << img_path << endl;
		return 0;
	}

	vector<KeyPoint> keypoints;
	if (fs::exists(yaml_path)) {
		FileStorage fs(yaml_path.c_str(), FileStorage::READ);
		read(fs["keypoints"], keypoints);
	} else {
		SiftFeatureDetector detector;
		detector.detect(current_img, keypoints);
		FileStorage fs(yaml_path.c_str(), FileStorage::WRITE);
		fs << "keypoints" << keypoints;
	}
	// print_keypoints(keypoints);

	// int fc = keypoints.size() * fc / 100;
	// fc = fc > 10 ? fc : 10;

	// not enough feature, dont use them
	if (keypoints.size() == 0) {
		cout << "Image " << img_path << " has no feature" << endl;
		return 0;
	}

	vector<KeyPoint> selected_keypoints;
	if (fc == 0) {
		selected_keypoints = keypoints;
	} else {
		select_keypoints(keypoints, selected_keypoints, fc);
	}

	cout << setw(50) << left << img_path
			<< setw(10) << right << keypoints.size()
			<< setw(10) << selected_keypoints.size() << endl;

	// extract descriptors
	Mat descriptors;
	SiftDescriptorExtractor extractor;
	extractor.compute(current_img, selected_keypoints, descriptors);

	// cout << img_path << ":" << endl;
	// print_keypoints(selected_keypoints);
	// cout << descriptors << endl;

	if (show_images) {
		Mat white = Mat::ones(current_img.rows, current_img.cols, CV_8UC1) * 255;
		Mat black = Mat::zeros(current_img.rows, current_img.cols, CV_8UC1);
		Mat img_density = Mat::zeros(current_img.rows, current_img.cols, CV_8UC1);
		Mat img(current_img);

		drawKeypoints(current_img, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("kp_on_img", img);

		drawKeypoints(black, keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("kp_on_black", img);
		
		drawKeypoints(black, selected_keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("selected_kp_on_white", img);

		calc_scale_density_img(current_img, keypoints, img_density);
		imshow("density", img_density);

		drawKeypoints(img_density, selected_keypoints, img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("selected_kp_on_density", img);

		imshow("img", current_img);
		waitKey(0);
	}

	descs = descriptors;
	kps = selected_keypoints;
	return 1;
}

void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int fc) {
	// filter keypoints
	filtered_keypoints.clear();
	if (filter_kp) {
		filter_keypoints(k, filtered_keypoints);
	} else {
		filtered_keypoints = k;
	}
	int filtered_kp_size = filtered_keypoints.size();

	if (sort_type == -5) { // scale + scale density
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_comparator);
		vector<KeyPoint> scale_selected;
		for (int i = 0; i < 2 * fc && i < filtered_kp_size; ++i) {
			scale_selected.push_back(filtered_keypoints[i]);
		}
		calc_scale_response_density_space(scale_selected);
		sort(scale_selected.begin(), scale_selected.end(), scale_density_comparator);
		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			s.push_back(scale_selected[i]);
		}
	} else if (sort_type == -4) { // scale density with substrating
		calc_scale_response_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);

		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			KeyPoint keypoint = filtered_keypoints[0];
			s.push_back(keypoint);

			// subtract the contribution of this keypoint
			for (int j = 0; j < filtered_kp_size; j++) {
				KeyPoint kp = filtered_keypoints[i];
				float dist = keypoint_dist(kp, keypoint);
				boost::math::normal_distribution<> norm_dist(0.0, 1 / keypoint.size);
				float density = boost::math::pdf(norm_dist, dist);
				int x = kp.pt.x;
				int y = kp.pt.y;
				scale_response_density_space[x][y] -= density;
			}

			// delete the selected keypoint
			filtered_keypoints.erase(filtered_keypoints.begin());
			// resort
			sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);
		}
	} else if (sort_type == -3) { // scale density with kdtree
		calc_scale_response_density_space(filtered_keypoints);

		// create keypoint map for easy deletion
		map<int, KeyPoint, scale_response_density_classcomp> mk;
		map<int, KeyPoint>::iterator it;

		// create kd-tree index
		Mat coord = Mat::zeros(filtered_kp_size, 2, CV_32F);
		for (int i = 0; i < filtered_kp_size; ++i) {
			mk[i] = filtered_keypoints[i];
			coord.ptr<float>(i)[0] = filtered_keypoints[i].pt.x;
			coord.ptr<float>(i)[1] = filtered_keypoints[i].pt.y;
		}
		
		flann::Index kdtree_index(coord, flann::KDTreeIndexParams(4));

		int max_count = 100;
		Mat indices = Mat::zeros(1, max_count, CV_32F);
		Mat dists = Mat::zeros(1, max_count, CV_32F);

		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			if (mk.size() == 0)
				break;
			
			KeyPoint query_keypoint = mk.begin()->second;

			s.push_back(query_keypoint);

			Mat query = Mat::zeros(1, 2, CV_32F);
			query.ptr<float>(0)[0] = query_keypoint.pt.x;
			query.ptr<float>(0)[1] = query_keypoint.pt.y;
			float radius = query_keypoint.size * query_keypoint.size / 4;
			int found_count = kdtree_index.radiusSearch(query, indices, dists, radius, max_count, flann::SearchParams());
			for (int j = 0; j < found_count; ++j) {
				mk.erase(indices.at<int>(j));
			}
		}
	} else if (sort_type == -2) { // scale density
		calc_scale_response_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);
		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == -1) { // scale
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_comparator);
		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 1) { // response
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), response_comparator);
		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 2) { // response density
		calc_reponse_response_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), density_comparator);
		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 3) { // response density with kdtree
		calc_reponse_response_density_space(filtered_keypoints);

		// create keypoint map for easy deletion
		map<int, KeyPoint, response_density_classcomp> mk;
		map<int, KeyPoint>::iterator it;

		// create kd-tree index
		Mat coord = Mat::zeros(filtered_kp_size, 2, CV_32F);
		for (int i = 0; i < filtered_kp_size; ++i) {
			mk[i] = filtered_keypoints[i];
			coord.ptr<float>(i)[0] = filtered_keypoints[i].pt.x;
			coord.ptr<float>(i)[1] = filtered_keypoints[i].pt.y;
		}
		
		flann::Index kdtree_index(coord, flann::KDTreeIndexParams(4));

		int max_count = 100;
		Mat indices = Mat::zeros(1, max_count, CV_32F);
		Mat dists = Mat::zeros(1, max_count, CV_32F);

		for (int i = 0; i < fc && i < filtered_kp_size; ++i) {
			if (mk.size() == 0)
				break;

			KeyPoint query_keypoint = mk.begin()->second;

			s.push_back(query_keypoint);

			Mat query = Mat::zeros(1, 2, CV_32F);
			query.ptr<float>(0)[0] = query_keypoint.pt.x;
			query.ptr<float>(0)[1] = query_keypoint.pt.y;
			float radius = query_keypoint.size * query_keypoint.size / 4;
			int found_count = kdtree_index.radiusSearch(query, indices, dists, radius, max_count, flann::SearchParams());
			for (int j = 0; j < found_count; ++j) {
				mk.erase(indices.at<int>(j));
			}
		}
	}
}

void filter_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& f) {
	if (sort_type < 0)
		sort(k.begin(), k.end(), scale_comparator);
	else
		sort(k.begin(), k.end(), response_comparator);

	for (int i = k.size() - 1; i >= 0; --i) {
		KeyPoint k1 = k[i];
		bool add = 1;
		for (int j = 0; j < f.size(); ++j) {
			KeyPoint k2 = f[j];
			float dist = keypoint_dist(k1, k2);
			if (dist < keypoint_filter_threshold) {
				add = 0;
				break;
			}
		}
		if (add)
			f.push_back(k1);
	}
}

void calc_response_density_img(const Mat& img, vector<KeyPoint> kp, Mat& ds) {
	int rows = img.rows;
	int cols = img.cols;
	int size = kp.size();
	Mat tmp = Mat::zeros(rows, cols, CV_32F);

	float max_value = 0, min_value = 9999999;
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				KeyPoint p = kp[k];
				float dist = sqrt(pow(i - p.pt.x, 2) + pow(j - p.pt.y, 2));
				boost::math::normal_distribution<> norm_dist(0.0, 1 / p.response * 100);
				sum += boost::math::pdf(norm_dist, dist);
			}
			tmp.ptr<float>(j)[i] = sum;
			if (sum > max_value)
				max_value = sum;
			if (sum < min_value)
				min_value = sum;
		}
	}

	ds = Mat(rows, cols, CV_8UC1);
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			ds.at<uchar>(j, i) = (tmp.at<float>(j, i) - min_value) / (max_value - min_value) * 255;
		}
	}

	equalizeHist(ds, ds);
}

void calc_reponse_response_density_space(const vector<KeyPoint>& kp) {
	int kp_size = kp.size();
	for (int i = 0; i < kp_size; i++) {
		KeyPoint ki = kp[i];
		float density = 0;
		for (int j = 0; j < kp_size; j++) {
			KeyPoint kj = kp[j];
			float dist = keypoint_dist(ki, kj);
			boost::math::normal_distribution<> norm_dist(0.0, 1 / kj.response * 100);
			density += boost::math::pdf(norm_dist, dist);
		}
		int x = ki.pt.x;
		int y = ki.pt.y;
		response_density_space[x][y] = density;
	}
}

void calc_scale_density_img(const Mat& img, vector<KeyPoint> kp, Mat& ds) {
	int rows = img.rows;
	int cols = img.cols;
	int size = kp.size();
	Mat tmp = Mat::zeros(rows, cols, CV_32F);

	float max_value = 0, min_value = 9999999;
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				KeyPoint p = kp[k];
				float dist = sqrt(pow(i - p.pt.x, 2) + pow(j - p.pt.y, 2));
				boost::math::normal_distribution<> norm_dist(0.0, 20 / p.size);
				sum += boost::math::pdf(norm_dist, dist);
			}
			tmp.ptr<float>(j)[i] = sum;
			if (sum > max_value)
				max_value = sum;
			if (sum < min_value)
				min_value = sum;
		}
	}

	ds = Mat(rows, cols, CV_8UC1);
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			ds.at<uchar>(j, i) = (tmp.at<float>(j, i) - min_value) / (max_value - min_value) * 255;
		}
	}

	equalizeHist(ds, ds);
}

void calc_scale_response_density_space(const vector<KeyPoint>& kp) {
	int kp_size = kp.size();
	for (int i = 0; i < kp_size; i++) {
		KeyPoint ki = kp[i];
		float density = 0;
		for (int j = 0; j < kp_size; j++) {
			KeyPoint kj = kp[j];
			float dist = keypoint_dist(ki, kj);
			boost::math::normal_distribution<> norm_dist(0.0, 20 / kj.size);
			density += boost::math::pdf(norm_dist, dist);
		}
		int x = ki.pt.x;
		int y = ki.pt.y;
		scale_response_density_space[x][y] = density;
	}
}

float keypoint_dist(const KeyPoint& k1, const KeyPoint& k2) {
	float dist = point_dist(k1.pt, k2.pt);
	return dist;
}

float point_dist(const Point2f& p1, const Point2f& p2) {
	float x_dist = p1.x - p2.x;
	float y_dist = p1.y - p2.y;
	float dist = sqrt(x_dist*x_dist + y_dist*y_dist);
	return dist;
}

bool response_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return p1.response > p2.response;
}

float get_density(const KeyPoint& k) {
	int x = k.pt.x;
	int y = k.pt.y;
	return response_density_space[x][y];
}

bool density_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return get_density(p1) > get_density(p2);
}

float get_scale_density(const KeyPoint& k) {
	int x = k.pt.x;
	int y = k.pt.y;
	return scale_response_density_space[x][y];
}

bool scale_density_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return get_scale_density(p1) > get_scale_density(p2);
}

bool scale_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return p1.size > p2.size;
}
