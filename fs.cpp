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
#include <boost/math/distributions/normal.hpp>

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

#define IMAGE_COUNT 10000
#define IMAGE_WIDTH 1000

// TODO: config
/* CONFIGURATIONS */
int keypoint_filter_threshold = 5;
bool filter_kp = 1;
int sort_type = 0;
bool use_fast_emd = 1;
float gd_dist = 200;
float radius = 10;
boost::math::normal_distribution<> norm_dist(0.0, 2);
/* END CONFIGURATIONS */

void help();
void find_images(const char *path);
void print_progress(time_t start, int total, int& completed, int per_cout);
void print_keypoints(const vector<KeyPoint>& kp);

void extract_descriptors(int index, int feature_count);
void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int fc);
void filter_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& f);
void calc_density_space_mat(const Mat& img, vector<KeyPoint> kp, Mat& ds);
void calc_density_space(const vector<KeyPoint>& kp);
void calc_scale_density_space(const vector<KeyPoint>& kp);
float keypoint_dist(const KeyPoint& k1, const KeyPoint& k2);
float point_dist(const Point2f& p1, const Point2f& p2);
bool response_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_density(const KeyPoint& k);
bool density_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_scale_density(const KeyPoint& k);
bool scale_density_comparator(const KeyPoint& p1, const KeyPoint& p2);
bool scale_comparator(const KeyPoint& p1, const KeyPoint& p2);

float calc_emd(int first, int second);
double calc_fast_emd(int first, int second);
float calc_scalar_manitude(const Mat& desc, int index);
void calc_contrast(const vector<KeyPoint>& kp);
void calc_dist_mat(const Mat& desc_1, const Mat& desc_2, bool gd = false);
float euclid_dist(const float *v1, const float *v2, int dim1, int dim2);
float emd_dist(feature_t *f1, feature_t *f2);
double fast_emd_dist(feature_tt *f1, feature_tt *f2);

Mat dist_mat;
Mat img_descriptors[IMAGE_COUNT];
vector<KeyPoint> img_keypoints[IMAGE_COUNT];
vector<string> img_paths;
bool use_image[IMAGE_COUNT];

float density_space[IMAGE_WIDTH][IMAGE_WIDTH];
float scale_density_space[IMAGE_WIDTH][IMAGE_WIDTH];
Mat current_img;

vector<KeyPoint> filtered_keypoints;
struct density_classcomp {
	bool operator()(const int& lhs, const int& rhs) const {
		KeyPoint kp1, kp2;
		kp1 = filtered_keypoints[lhs];
		kp2 = filtered_keypoints[rhs];
		return density_comparator(kp1, kp2);
	}
};
struct scale_density_classcomp {
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

	int feature_count = atoi(argv[1]);
	sort_type = atoi(argv[2]);
	const char *img_path = argv[3];
	const char *out_path = argv[4];
	if (argc > 5)
		gd_dist = atof(argv[5]);
	if (argc > 6)
		radius = atof(argv[6]);

	// finds all png images and stores paths of founds at img_paths
	find_images(img_path);
	int total_image = img_paths.size();

	int total_process = total_image;
	int completed_process = 0;
	time_t start = time(0);

	cout << "Calculating feature descriptors of images" << endl;
	// calculate graphs of all images
	for (int i = 0; i < total_image; ++i)
		extract_descriptors(i, feature_count);

	cout << "Calculating emd distance of images" << endl;

	total_process = (total_image + 1) * (total_image) / 2;
	completed_process = 0;
	start = time(0);

	string img_path_1, img_path_2;

	ofstream ofile(out_path);

	tictoc timer;
	timer.tic();
	// calculate emd distances of each pair of images
	for (int i = 0; i < total_image; ++i) {
		if (!use_image[i])
			continue;
		img_path_1 = img_paths[i];
		for (int j = i + 1; j < total_image; ++j) {
			if (!use_image[j])
				continue;
			img_path_2 = img_paths[j];

			double emd_dist;
			if (use_fast_emd)
				emd_dist = calc_fast_emd(i, j);
			else
				emd_dist = calc_emd(i, j);
			
			// cout << setprecision(3) << fixed;
			// timer.clear();
			// timer.tic();
			// emd_dist = calc_fast_emd(i, j);
			// timer.toc();
			// cout << "fast: " << timer.totalTimeSec() << endl;
			// cout << img_path_1 << " " << img_path_2 << " " << emd_dist << endl;
			
			// timer.clear();
			// timer.tic();
			// emd_dist = calc_emd(i, j);
			// timer.toc();
			// cout << "rubner: " << timer.totalTimeSec() << endl;
			// cout << img_path_1 << " " << img_path_2 << " " << emd_dist << endl;

			ofile << setprecision(3) << fixed;
			ofile << img_path_1 << " " << img_path_2 << " " << emd_dist << endl;

			print_progress(start, total_process, completed_process, 20);
		}
	}
	timer.toc();
	cout << "Time in seconds: " << timer.totalTimeSec() << endl;

	ofile.close();
}

void help() {
	cout << "Usage: ./feature-selection <feature_count> <sort_type> <img_path> <out_path> <fast_thresh>" << endl;
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
	for (int i = 0; i < (int) kp.size(); ++i) {
		k = kp[i];
		cout << "(" << k.pt.x << "," << k.pt.y << "): " << k.response << ", " << k.size << endl;
	}
}

void find_images(const char *path) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(path)) == NULL) {
		cout << "--(!) Could not open directory: " << path << endl;
		return;
	}

	while ((ent = readdir(dir)) != NULL) {
		if (strcmp(".", ent->d_name) == 0 || strcmp("..", ent->d_name) == 0)
			continue;

		fs::path dir_path(path);
		fs::path file_path(ent->d_name);
		fs::path img_path = dir_path / file_path;
		if (ent->d_type == 4) {
			find_images(img_path.c_str());
		} else if (ent->d_type == 8) {
			if (!boost::algorithm::ends_with(img_path.c_str(), ".png"))
				continue;
			// cout << img_path << endl;
			img_paths.push_back(img_path.c_str());
		}
		sort(img_paths.begin(), img_paths.end());
	}
	closedir(dir);
}

void extract_descriptors(int index, int feature_count) {
	string img_path = img_paths[index];
	// cout << "Calc img matrix: " << img_path << endl;
	current_img = imread(img_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (!current_img.data) {
		cout << "--(!) Error reading image: " << img_path << endl;
	}

	vector<KeyPoint> keypoints;
	SiftFeatureDetector detector;
	detector.detect(current_img, keypoints);
	// print_keypoints(keypoints);

//	int fc = keypoints.size() * feature_count / 100;
//	feature_count = fc > 10 ? fc : 10;

	// not enough feature, dont use them
	if (keypoints.size() < feature_count) {
		use_image[index] = 0;
		cout << "Image " << img_path << " has not enough feature" << endl;
		return;
	} else {
		use_image[index] = 1;
	}

	vector<KeyPoint> selected_keypoints;
	if (feature_count == 0) {
		selected_keypoints = keypoints;
	} else {
		select_keypoints(keypoints, selected_keypoints, feature_count);
	}

	cout << setw(50) << left << img_path
			<< setw(10) << right << keypoints.size()
			<< setw(10) << selected_keypoints.size() << endl;

	// TODO: KD-TREE
//	Mat coord = Mat::zeros(selected_keypoints.size(), 2, CV_32F);
//	for (int i = 0; i < (int) selected_keypoints.size(); ++i) {
//		coord.ptr<float>(i)[0] = selected_keypoints[i].pt.x;
//		coord.ptr<float>(i)[1] = selected_keypoints[i].pt.y;
//	}
//	flann::Index kdtree_index(coord, flann::KDTreeIndexParams());
//
//	int max_count = 100;
//	Mat indices = Mat::zeros(1, max_count, CV_32F);
//	Mat dists = Mat::zeros(1, max_count, CV_32F);
//	float radius = 100;
//
//	int found_count = kdtree_index.radiusSearch(coord.rowRange(0, 1), indices,
//			dists, radius, max_count, flann::SearchParams());
//
//	cout << "keypoint size: " << endl;
//	cout << coord.rowRange(0, 1) << endl;
//	cout << "indices:" << endl;
//	for (int i = 0; i < found_count; ++i) {
//		cout << indices.at<int>(i) << endl;
//	}
//	cout << "dists:" << endl;
//	cout << dists << endl;
//
//	vector<KeyPoint> selected_keypoints2;
//
//	for (int i = 0; i < found_count; ++i) {
//		selected_keypoints2.push_back(selected_keypoints[indices.at<int>(i)]);
//	}
//
//	selected_keypoints = selected_keypoints2;
//
//	print_keypoints(selected_keypoints);
	//#########################################################################

	// extract descriptors
	Mat descriptors;
	SiftDescriptorExtractor extractor;
	extractor.compute(current_img, selected_keypoints, descriptors);

	// cout << img_path << ":" << endl;
	// print_keypoints(selected_keypoints);
	// cout << descriptors << endl;

//	print_keypoints(selected_keypoints);
//	calc_scalar_manitude(descriptors, keypoints);
//	sort(keypoints.begin(), keypoints.end(), scalar_comparator);
//	selected_keypoints.clear();
//	for (int i = 0; i < 5; ++i)
//		selected_keypoints.push_back(keypoints[i]);
//	extractor->compute(current_img, selected_keypoints, descriptors);

	// Mat img_keypoints(current_img);
	// // Mat white = Mat::zeros(current_img.rows, current_img.cols, CV_8U);
	// // drawKeypoints(white, selected_keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// drawKeypoints(current_img, selected_keypoints, img_keypoints,
	// 		Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// imshow(img_path, img_keypoints);
	// waitKey(0);
//
//	Mat img_density;
//	calc_density_space_mat(current_img, keypoints, img_density);
//	cvtColor(img_density, img_density, CV_RGB2GRAY);
//	Mat img_density_with_keypoints(img_density);
//	drawKeypoints(img_density, selected_keypoints, img_density_with_keypoints,
//			Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	imshow(img_path, img_density);
//
//	waitKey(0);

	img_descriptors[index] = descriptors;
	img_keypoints[index] = selected_keypoints;
}

void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int feature_count) {
	// filter keypoints
	filtered_keypoints.clear();
	if (filter_kp) {
		filter_keypoints(k, filtered_keypoints);
	} else {
		filtered_keypoints = k;
	}
	int filtered_kp_size = filtered_keypoints.size();

	if (sort_type == -4) { // scale density with substrating
		calc_scale_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);

		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			KeyPoint keypoint = filtered_keypoints[0];
			s.push_back(keypoint);

			// subtract the contribution of this keypoint
			for (int j = 0; j < filtered_kp_size; j++) {
				KeyPoint kp = filtered_keypoints[i];
				float dist = keypoint_dist(kp, keypoint);
				float density = boost::math::pdf(norm_dist, dist) * keypoint.size;
				int x = kp.pt.x;
				int y = kp.pt.y;
				scale_density_space[x][y] -= density;
			}

			// delete the selected keypoint
			filtered_keypoints.erase(filtered_keypoints.begin());
			// resort
			sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);
		}
	} else if (sort_type == -3) { // scale density with kdtree
		calc_scale_density_space(filtered_keypoints);

		// create keypoint map for easy deletion
		map<int, KeyPoint, scale_density_classcomp> mk;
		map<int, KeyPoint>::iterator it;

		for (int i = 0; i < filtered_kp_size; ++i) {
			mk[i] = filtered_keypoints[i];
		}

		// create kd-tree index
		Mat coord = Mat::zeros(mk.size(), 2, CV_32F);
		for (it = mk.begin(); it != mk.end(); ++it) {
			coord.ptr<float>(it->first)[0] = it->second.pt.x;
			coord.ptr<float>(it->first)[1] = it->second.pt.y;
		}
		flann::Index kdtree_index(coord, flann::KDTreeIndexParams());

		int max_count = 100;
		Mat indices = Mat::zeros(1, max_count, CV_32F);
		Mat dists = Mat::zeros(1, max_count, CV_32F);

		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			KeyPoint query_keypoint = mk.begin()->second;

			s.push_back(query_keypoint);

			Mat query = Mat::zeros(1, 2, CV_32F);
			query.ptr<float>(0)[0] = query_keypoint.pt.x;
			query.ptr<float>(0)[1] = query_keypoint.pt.y;

			int found_count = kdtree_index.radiusSearch(query, indices, dists, radius, max_count, flann::SearchParams());
			for (int j = 0; j < found_count; ++j) {
				mk.erase(indices.at<int>(j));
			}
		}
	} else if (sort_type == -2) { // scale density
		calc_scale_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_density_comparator);
		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == -1) { // scale
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), scale_comparator);
		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 1) { // response
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), response_comparator);
		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 2) { // response density
		calc_density_space(filtered_keypoints);
		sort(filtered_keypoints.begin(), filtered_keypoints.end(), density_comparator);
		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			s.push_back(filtered_keypoints[i]);
		}
	} else if (sort_type == 3) { // response density with kdtree
		calc_density_space(filtered_keypoints);

		// create keypoint map for easy deletion
		map<int, KeyPoint, density_classcomp> mk;
		map<int, KeyPoint>::iterator it;

		for (int i = 0; i < filtered_kp_size; ++i) {
			mk[i] = filtered_keypoints[i];
		}

		// create kd-tree index
		Mat coord = Mat::zeros(mk.size(), 2, CV_32F);
		for (it = mk.begin(); it != mk.end(); ++it) {
//			cout << it->first << "\t" << get_density(it->second) << endl;
			coord.ptr<float>(it->first)[0] = it->second.pt.x;
			coord.ptr<float>(it->first)[1] = it->second.pt.y;
		}
		flann::Index kdtree_index(coord, flann::KDTreeIndexParams());

		int max_count = 100;
		Mat indices = Mat::zeros(1, max_count, CV_32F);
		Mat dists = Mat::zeros(1, max_count, CV_32F);

		for (int i = 0; i < feature_count && i < filtered_kp_size; ++i) {
			KeyPoint query_keypoint = mk.begin()->second;

			s.push_back(query_keypoint);

			Mat query = Mat::zeros(1, 2, CV_32F);
			query.ptr<float>(0)[0] = query_keypoint.pt.x;
			query.ptr<float>(0)[1] = query_keypoint.pt.y;

			int found_count = kdtree_index.radiusSearch(query, indices, dists,
					radius, max_count, flann::SearchParams());

//			cout << "me:" << mk.begin()->first << endl;
//			cout << "found_count: " << found_count << endl;
//			cout << "indices:" << endl;
//			cout << indices << endl;
//			cout << "dists:" << endl;
//			cout << dists << endl;

//			cout << "mk size before: " << mk.size() << endl;
			for (int j = 0; j < found_count; ++j) {
				mk.erase(indices.at<int>(j));
			}
//			cout << "mk size after: " << mk.size() << endl;
//			cout << endl << endl;
		}
	}
}

void filter_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& f) {
	if (sort_type < 0)
		sort(k.begin(), k.end(), scale_comparator);
	else
		sort(k.begin(), k.end(), response_comparator);

	for (int i = k.size() - 1; i > 0; --i) {
		KeyPoint k1 = k[i];
		bool add = 1;
		for (int j = i - 1; j >= 0; --j) {
			KeyPoint k2 = k[j];
			float dist = keypoint_dist(k1, k2);
			if (dist < keypoint_filter_threshold) {
				add = 0;
				break;
			}
		}
		if (add)
			f.push_back(k1);
	}
	reverse(f.begin(), f.end());
}

void calc_density_space_mat(const Mat& img, vector<KeyPoint> kp, Mat& ds) {
	int rows = img.rows;
	int cols = img.cols;
	int size = kp.size();
	ds = Mat::zeros(rows, cols, CV_32F);

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				KeyPoint p = kp[k];
				float dist = sqrt(pow(i - p.pt.x, 2) + pow(j - p.pt.y, 2));
				sum += boost::math::pdf(norm_dist, dist) * p.response * 10;
			}
			ds.ptr<float>(j)[i] = sum;
		}
	}
}

void calc_density_space(const vector<KeyPoint>& kp) {
	int kp_size = kp.size();
	for (int i = 0; i < kp_size; i++) {
		KeyPoint ki = kp[i];
		float density = 0;
		for (int j = 0; j < kp_size; j++) {
			KeyPoint kj = kp[j];
			float dist = keypoint_dist(ki, kj);
			density += boost::math::pdf(norm_dist, dist) * kj.response * 100;
		}
		int x = ki.pt.x;
		int y = ki.pt.y;
		density_space[x][y] = density;
	}
}

void calc_scale_density_space(const vector<KeyPoint>& kp) {
	int kp_size = kp.size();
	for (int i = 0; i < kp_size; i++) {
		KeyPoint ki = kp[i];
		float density = 0;
		for (int j = 0; j < kp_size; j++) {
			KeyPoint kj = kp[j];
			float dist = keypoint_dist(ki, kj);
			density += boost::math::pdf(norm_dist, dist) * kj.size;
		}
		int x = ki.pt.x;
		int y = ki.pt.y;
		scale_density_space[x][y] = density;
	}
}

float calc_emd(int first, int second) {
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat desc_1, desc_2;

	keypoints_1 = img_keypoints[first];
	desc_1 = img_descriptors[first];

	keypoints_2 = img_keypoints[second];
	desc_2 = img_descriptors[second];

	calc_dist_mat(desc_1, desc_2);

	int feature_count_1 = keypoints_1.size();
	int feature_count_2 = keypoints_2.size();

	signature_t signature_1;
	signature_t signature_2;

	signature_1.n = feature_count_1;
	signature_2.n = feature_count_2;

	signature_1.Features = new feature_t[feature_count_1];
	signature_2.Features = new feature_t[feature_count_2];

	signature_1.Weights = new float[feature_count_1];
	signature_2.Weights = new float[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		if (sort_type < 0) {
			// signature_1.Weights[i] = keypoints_1.at(i).size;
			signature_1.Weights[i] = calc_scalar_manitude(desc_1, i);
		} else {
			signature_1.Weights[i] = keypoints_1.at(i).response * 100;
		}

		signature_1.Features[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		if (sort_type < 0) {
			// signature_2.Weights[i] = keypoints_2.at(i).size;
			signature_2.Weights[i] = calc_scalar_manitude(desc_2, i);
		} else {
			signature_2.Weights[i] = keypoints_2.at(i).response * 100;
		}

		signature_2.Features[i] = i;
	}

	float dist = emd(&signature_1, &signature_2, emd_dist, 0, 0);

	delete[] signature_1.Features;
	delete[] signature_1.Weights;
	delete[] signature_2.Features;
	delete[] signature_2.Weights;

	// cout << "EMD: " << dist << endl;

	return dist;
}

double calc_fast_emd(int first, int second) {
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat desc_1, desc_2;

	keypoints_1 = img_keypoints[first];
	desc_1 = img_descriptors[first];

	keypoints_2 = img_keypoints[second];
	desc_2 = img_descriptors[second];

	calc_dist_mat(desc_1, desc_2, true);

	int feature_count_1 = keypoints_1.size();
	int feature_count_2 = keypoints_2.size();

	signature_tt<double> signature_1;
	signature_tt<double> signature_2;

	signature_1.n = feature_count_1;
	signature_2.n = feature_count_2;

	signature_1.Features = new feature_tt[feature_count_1];
	signature_2.Features = new feature_tt[feature_count_2];

	signature_1.Weights = new double[feature_count_1];
	signature_2.Weights = new double[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		if (sort_type < 0) {
			// signature_1.Weights[i] = keypoints_1.at(i).size;
			signature_1.Weights[i] = calc_scalar_manitude(desc_1, i);
		} else {
			signature_1.Weights[i] = keypoints_1.at(i).response * 100;
		}

		signature_1.Features[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		if (sort_type < 0) {
			// signature_2.Weights[i] = keypoints_2.at(i).size;
			signature_2.Weights[i] = calc_scalar_manitude(desc_2, i);
		} else {
			signature_2.Weights[i] = keypoints_2.at(i).response * 100;
		}

		signature_2.Features[i] = i;
	}

	double dist = emd_hat_signature_interface<double>(&signature_1, &signature_2, fast_emd_dist, -1);

	delete[] signature_1.Features;
	delete[] signature_1.Weights;
	delete[] signature_2.Features;
	delete[] signature_2.Weights;

	// cout << "EMD: " << dist << endl;

	return dist / 1000;
}

float calc_scalar_manitude(const Mat& desc, int index) {
	const float* d = desc.ptr<float>(index);
	int cols = desc.cols;
	float magnitude = 0;
	for (int i = 0; i < cols; ++i)
		magnitude += d[i] * d[i];
	return sqrt(magnitude);
}

void calc_dist_mat(const Mat& desc_1, const Mat& desc_2, bool gd) {
	float dist;
	int rows = desc_1.rows;
	int cols = desc_2.rows;
	int dim1 = desc_1.cols;
	int dim2 = desc_2.cols;
	dist_mat = Mat::zeros(rows, cols, CV_32F);
	for (int i = 0; i < rows; ++i) {
		float *row = dist_mat.ptr<float>(i);
		const float *v1 = desc_1.ptr<float>(i);
		for (int j = 0; j < cols; ++j) {
			const float *v2 = desc_2.ptr<float>(j);
			dist = euclid_dist(v1, v2, dim1, dim2);
			if (gd)
				row[j] = gd_dist < dist ? gd_dist : dist;
			else
				row[j] = dist;
		}
	}
}

float euclid_dist(const float *v1, const float *v2, int dim1, int dim2) {
	int max_dim = max(dim1, dim2);
	float d1, d2;
	float dist = 0;
	for (int i = 0; i < max_dim; ++i) {
		if (i < dim1)
			d1 = v1[i];
		else
			d1 = 0;

		if (i < dim2)
			d2 = v2[i];
		else
			d2 = 0;

		dist += pow(d1 - d2, 2);
	}
	return sqrt(dist);
}

float emd_dist(feature_t *f1, feature_t *f2) {
	float dist = dist_mat.at<float>(*f1, *f2);
	return dist;
}

double fast_emd_dist(feature_tt *f1, feature_tt *f2) {
	double dist = dist_mat.at<float>(*f1, *f2);
	return dist;
}

float keypoint_dist(const KeyPoint& k1, const KeyPoint& k2) {
	float dist = point_dist(k1.pt, k2.pt);
	return dist;
}

float point_dist(const Point2f& p1, const Point2f& p2) {
	float x_dist = p1.x - p2.x;
	float y_dist = p1.y - p2.y;
	float dist;
	dist = sqrt(pow(x_dist, 2) + pow(y_dist, 2));
	return dist;
}

bool response_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return p1.response > p2.response;
}

float get_density(const KeyPoint& k) {
	int x = k.pt.x;
	int y = k.pt.y;
	return density_space[x][y];
}

bool density_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return get_density(p1) > get_density(p2);
}

float get_scale_density(const KeyPoint& k) {
	int x = k.pt.x;
	int y = k.pt.y;
	return scale_density_space[x][y];
}

bool scale_density_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return get_scale_density(p1) > get_scale_density(p2);
}

bool scale_comparator(const KeyPoint& p1, const KeyPoint& p2) {
	return p1.size > p2.size;
}
