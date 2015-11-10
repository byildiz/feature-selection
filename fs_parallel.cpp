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

#define IMAGE_WIDTH 1000

/* CONFIGURATIONS */
int keypoint_filter_threshold = 5;
bool filter_kp = 1;
int sort_type = 0;
bool use_fast_emd = 1;
double gd_dist = -1;
float radius = 10;
boost::math::normal_distribution<> norm_dist(0.0, 2);
/* END CONFIGURATIONS */

void help();
void print_keypoints(const vector<KeyPoint>& kp);
double calc_emd(int first, int second);
double calc_fast_emd(int first, int second);
void calc_density_space_mat(const Mat& img, vector<KeyPoint> kp, Mat& ds);
void calc_density_space(const vector<KeyPoint>& kp);
void calc_scale_density_space(const vector<KeyPoint>& kp);
void calc_desc_dist(const Mat& desc_1, const Mat& desc_2);
void extract_descriptors(int index, int count);
double euclid_dist(const float *v1, const float *v2, int dim1, int dim2);
float emd_dist(feature_t *f1, feature_t *f2);
double fast_emd_dist(feature_tt *f1, feature_tt *f2);
float keypoint_dist(const KeyPoint& k1, const KeyPoint& k2);
float point_dist(const Point2f& p1, const Point2f& p2);
bool response_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_density(const KeyPoint& k);
bool density_comparator(const KeyPoint& p1, const KeyPoint& p2);
float get_scale_density(const KeyPoint& k);
bool scale_density_comparator(const KeyPoint& p1, const KeyPoint& p2);
bool scale_comparator(const KeyPoint& p1, const KeyPoint& p2);
void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int fc);
void filter_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& f);
void cluster_keypoints(const vector<KeyPoint>& keypoints, Mat& centers, Mat& labels, int cluster_count);

Mat current_img;
char *img_paths[2];
Mat img_descriptors[2];
vector<KeyPoint> img_keypoints[2];
float density_space[IMAGE_WIDTH][IMAGE_WIDTH];
float scale_density_space[IMAGE_WIDTH][IMAGE_WIDTH];
Mat dist_mat;

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

SiftFeatureDetector detector;
SiftDescriptorExtractor extractor;

int main(int argc, char *argv[]) {
	if (argc != 4) {
		help();
		return -1;
	}

	const char *config_path = argv[1];
	img_paths[0] = argv[2];
	img_paths[1] = argv[3];

	ifstream config_file(config_path);
	string x;
	int feature_count;
	config_file >> x >> feature_count;
	config_file >> x >> sort_type;
	config_file >> x >> gd_dist;
	config_file >> x >> radius;

	extract_descriptors(0, feature_count);
	extract_descriptors(1, feature_count);

	if (img_keypoints[0].size() < feature_count || img_keypoints[1].size() < feature_count)
		return 1;

	tictoc timer;
	timer.tic();

	double emd_dist;
	if (use_fast_emd)
		emd_dist = calc_fast_emd(0, 1);
	else
		emd_dist = calc_emd(0, 1);

	cout << setprecision(3) << fixed;
	cout << img_paths[0] << " " << img_paths[1] << " " << emd_dist << endl;

	timer.toc();
	cout << "Time in seconds: " << timer.totalTimeSec() << endl;
}

void help() {
	cout << "Usage: ./feature-selection <config_path> <img1_path> <img2_path>" << endl;
}

void print_keypoints(const vector<KeyPoint>& kp) {
	KeyPoint k;
	for (int i = 0; i < (int) kp.size(); ++i) {
		k = kp[i];
		cout << "(" << k.pt.x << "," << k.pt.y << "): " << k.response << ", " << k.size << endl;
	}
}

double calc_emd(int first, int second) {
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat desc_1, desc_2;

	keypoints_1 = img_keypoints[first];
	desc_1 = img_descriptors[first];

	keypoints_2 = img_keypoints[second];
	desc_2 = img_descriptors[second];

	calc_desc_dist(desc_1, desc_2);

	// cout << "(" << first << "," << second << ") feature distance:" << endl;
	// cout << feature_dist << endl;

	int feature_count_1 = keypoints_1.size();
	int feature_count_2 = keypoints_2.size();

	// cout << "Keypoint size 1: " << feature_count_1 << endl;
	// cout << "Keypoint size 2: " << feature_count_2 << endl;

	feature_t feature_1[feature_count_1];
	feature_t feature_2[feature_count_2];

	float weights_1[feature_count_1];
	float weights_2[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		if (sort_type < 0) {
			weights_1[i] = keypoints_1.at(i).size;
		} else {
			weights_1[i] = keypoints_1.at(i).response * 100;
		}

		feature_1[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		if (sort_type < 0) {
			weights_2[i] = keypoints_2.at(i).size;
		} else {
			weights_2[i] = keypoints_2.at(i).response * 100;
		}

		feature_2[i] = i;
	}

	signature_t signature_1 = { feature_count_1, feature_1, weights_1 };
	signature_t signature_2 = { feature_count_2, feature_2, weights_2 };

	double dist = emd(&signature_1, &signature_2, emd_dist, 0, 0);

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

	calc_desc_dist(desc_1, desc_2);

	// cout << "(" << first << "," << second << ") feature distance:" << endl;
	// cout << feature_dist << endl;

	int feature_count_1 = keypoints_1.size();
	int feature_count_2 = keypoints_2.size();

	// cout << "Keypoint size 1: " << feature_count_1 << endl;
	// cout << "Keypoint size 2: " << feature_count_2 << endl;

	feature_tt feature_1[feature_count_1];
	feature_tt feature_2[feature_count_2];

	double weights_1[feature_count_1];
	double weights_2[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		if (sort_type < 0) {
			weights_1[i] = keypoints_1.at(i).size;
		} else {
			weights_1[i] = keypoints_1.at(i).response * 100;
		}

		feature_1[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		if (sort_type < 0) {
			weights_2[i] = keypoints_2.at(i).size;
		} else {
			weights_2[i] = keypoints_2.at(i).response * 100;
		}

		feature_2[i] = i;
	}

	signature_tt<double> signature_1 = { feature_count_1, feature_1, weights_1 };
	signature_tt<double> signature_2 = { feature_count_2, feature_2, weights_2 };

	double dist = emd_hat_signature_interface<double>(&signature_1, &signature_2, fast_emd_dist, gd_dist);

	// cout << "EMD: " << dist << endl;

	return dist/1000.0;
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

void extract_descriptors(int index, int count) {

	char *img_path = img_paths[index];

	current_img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
	if (!current_img.data) {
		cout << "--(!) Error reading image: " << img_path << endl;
	}

	vector<KeyPoint> keypoints;
	detector.detect(current_img, keypoints);
	// print_keypoints(keypoints);

	if (keypoints.size() < count)
		return;

	vector<KeyPoint> selected_keypoints;
	if (count == 0) {
		selected_keypoints = keypoints;
	} else {
		select_keypoints(keypoints, selected_keypoints, count);
	}

	cout << setw(50) << left << img_path
			<< setw(10) << right << keypoints.size()
			<< setw(10) << selected_keypoints.size() << endl;

	// extract descriptors
	Mat descriptors;
	extractor.compute(current_img, selected_keypoints, descriptors);

	// print_keypoints(selected_keypoints);
	// calc_scalar_manitude(descriptors, keypoints);
	// sort(keypoints.begin(), keypoints.end(), scalar_comparator);
	// selected_keypoints.clear();
	// for (int i = 0; i < 5; ++i)
	// 	selected_keypoints.push_back(keypoints[i]);
	// extractor->compute(current_img, selected_keypoints, descriptors);

	// Mat img_keypoints(current_img);
	// // Mat white = Mat::zeros(current_img.rows, current_img.cols, CV_8U);
	// // drawKeypoints(white, selected_keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// drawKeypoints(current_img, selected_keypoints, img_keypoints,
	// 		Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// imshow(img_path, img_keypoints);
	// waitKey(0);

	// Mat img_density;
	// calc_density_space_mat(current_img, keypoints, img_density);
	// cvtColor(img_density, img_density, CV_RGB2GRAY);
	// Mat img_density_with_keypoints(img_density);
	// drawKeypoints(img_density, selected_keypoints, img_density_with_keypoints,
	// 		Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// imshow(img_path, img_density);

	// waitKey(0);

	img_keypoints[index] = selected_keypoints;
	img_descriptors[index] = descriptors;
}

void calc_desc_dist(const Mat& desc_1, const Mat& desc_2) {
	double dist;
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
			if (gd_dist == -1)
				row[j] = dist;
			else	
				row[j] = gd_dist < dist ? gd_dist : dist;
		}
	}
}

double euclid_dist(const float *v1, const float *v2, int dim1, int dim2) {
	int max_dim = max(dim1, dim2);
	float d1, d2;
	double dist = 0;
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

void select_keypoints(vector<KeyPoint>& k, vector<KeyPoint>& s, int feature_count) {
	// filter keypoints
	filtered_keypoints.clear();
	if (filter_kp) {
		filter_keypoints(k, filtered_keypoints);
	} else {
		filtered_keypoints = k;
	}
	int filtered_kp_size = filtered_keypoints.size();

//	cout << "features" << endl;
//	for (int i = 0; i < filtered_kp_size; ++i) {
//		cout << f[i].response << endl;
//	}
//	cout << endl;

//	if (use_kmeans) {
//		vector<Point2f> center_points;
//		// cluster keypoints
//		int cluster_count = min(feature_count, filtered_kp_size);
//		// cout << "Cluster count: " << cluster_count << endl;
//		Mat labels, centers;
//		cluster_keypoints(filtered_keypoints, centers, labels, cluster_count);
//
//		// plot clustered keypoints
//		Mat clustered_image(current_img);
//		RNG rng(12345);
//		for (int i = 0; i < cluster_count; ++i) {
//			Scalar s(rng.uniform(0, 255), rng.uniform(0, 255),
//					rng.uniform(0, 255));
//			vector<KeyPoint> cluster;
//			for (int j = 0; j < filtered_kp_size; ++j) {
//				int clusterIdx = labels.at<int>(j);
//				if (clusterIdx == i)
//					cluster.push_back(filtered_keypoints[j]);
//			}
//			drawKeypoints(clustered_image, cluster, clustered_image, s);
//		}
//		// imshow(img_path, clustered_image);
//
//		for (int i = 0; i < cluster_count; ++i) {
//			center_points.push_back(centers.at<Point2f>(i));
//		}
//	}

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

void cluster_keypoints(const vector<KeyPoint>& keypoints, Mat& centers,
		Mat& labels, int cluster_count) {
	Mat points(keypoints.size(), 1, CV_32FC2);
	for (int i = 0; i < (int) keypoints.size(); ++i) {
		points.ptr<Point2f>(i)[0] = keypoints[i].pt;
	}
	centers = Mat::zeros(cluster_count, 1, points.type());
	kmeans(points, cluster_count, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3,
			KMEANS_PP_CENTERS, centers);
}