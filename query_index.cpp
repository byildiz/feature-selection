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

// TODO: config
/* CONFIGURATIONS */
bool use_fast_emd = 1;
float gd_dist = 350;
/* END CONFIGURATIONS */

void help();
void print_progress(time_t start, int total, int& completed, int per_cout);
void print_keypoints(const vector<KeyPoint>& kp);

float calc_emd(int first, int second);
double calc_fast_emd(int first, int second);
float calc_scalar_manitude(const Mat& desc, int index);
void calc_contrast(const vector<KeyPoint>& kp);
void calc_dist_mat(const Mat& desc_1, const Mat& desc_2, bool gd = false);
float euclid_dist(const float *v1, const float *v2, int dim1, int dim2);
float emd_dist(feature_t *f1, feature_t *f2);
double fast_emd_dist(feature_tt *f1, feature_tt *f2);

Mat dist_mat;
vector<Mat> img_descriptors;
vector<vector<KeyPoint> > img_keypoints;
vector<string> img_paths;

int main(int argc, char *argv[]) {
	if (argc < 4) {
		help();
		return -1;
	}

	const char *index_path = argv[1];
	const char *queries_path = argv[2];
	const char *out_path = argv[3];
	if (argc > 5)
		gd_dist = atof(argv[5]);

	cout << "Reading queries from " << queries_path << endl;
	vector<string> query_paths;
	string query_path;
	int query_count;
	ifstream queries_file(queries_path);
	queries_file >> query_count;
	for (int i = 0; i < query_count; ++i) {
		queries_file >> query_path;
		query_paths.push_back(query_path);
	}

	cout << "Reading index from " << index_path << endl;
	FileStorage fs(index_path, FileStorage::READ);
	FileNode imgs = fs["index"];
	FileNodeIterator it;
	int img_count = 0;
	vector<pair<int, int> > query_ids;
	for (it = imgs.begin(); it != imgs.end(); ++it, ++img_count) {
		string img_path;
		vector<KeyPoint> img_keypoint;
		Mat img_descriptor;

		(*it)["path"] >> img_path;
		read((*it)["keypoints"], img_keypoint);
		(*it)["descriptors"] >> img_descriptor;

		img_paths.push_back(img_path);
		img_keypoints.push_back(img_keypoint);
		img_descriptors.push_back(img_descriptor);

		for (int i = 0; i < query_count; ++i) {
			if (img_path == query_paths[i])
				query_ids.push_back(pair<int, int>(img_count, i));
		}
	}

	query_count = query_ids.size();

	cout << "Calcing emds..." << endl;

	int total_process = query_count * img_count;
	int completed_process = 0;
	time_t start = time(0);

	string img_path_1, img_path_2;
	int query_id;

	ofstream ofile(out_path);

	tictoc timer;
	timer.tic();
	// calculate emd distances of each pair of images
	for (int i = 0; i < query_count; ++i) {
		query_id = query_ids[i].first;
		img_path_1 = query_paths[query_ids[i].second];

		for (int j = 0; j < img_count; ++j) {
			if (query_id == j)
				continue;

			img_path_2 = img_paths[j];

			double emd_dist;
			if (use_fast_emd)
				emd_dist = calc_fast_emd(query_id, j);
			else
				emd_dist = calc_emd(query_id, j);
			
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

			// print_progress(start, total_process, completed_process, 20);
		}
	}
	timer.toc();
	cout << "Time in seconds: " << timer.totalTimeSec() << endl;

	ofile.close();

	return 0;
}

void help() {
	cout << "Usage: ./feature-selection <index_path> <queries_path> <out_path> <fast_thresh=350>" << endl;
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
	cout << "Remaining time: ";
	cout << remaining_time / 60 << "min" << endl;
}

void print_keypoints(const vector<KeyPoint>& kp) {
	KeyPoint k;
	for (int i = 0; i < (int) kp.size(); ++i) {
		k = kp[i];
		cout << "(" << k.pt.x << "," << k.pt.y << "): " << k.response << ", " << k.size << endl;
	}
}

float calc_emd(int first, int second) {
	vector<KeyPoint> *keypoints_1, *keypoints_2;
	Mat *desc_1, *desc_2;

	keypoints_1 = &img_keypoints[first];
	desc_1 = &img_descriptors[first];

	keypoints_2 = &img_keypoints[second];
	desc_2 = &img_descriptors[second];

	calc_dist_mat(*desc_1, *desc_2);

	int feature_count_1 = keypoints_1->size();
	int feature_count_2 = keypoints_2->size();

	signature_t signature_1;
	signature_t signature_2;

	signature_1.n = feature_count_1;
	signature_2.n = feature_count_2;

	signature_1.Features = new feature_t[feature_count_1];
	signature_2.Features = new feature_t[feature_count_2];

	signature_1.Weights = new float[feature_count_1];
	signature_2.Weights = new float[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		signature_1.Weights[i] = calc_scalar_manitude(*desc_1, i);
		signature_1.Features[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		signature_2.Weights[i] = calc_scalar_manitude(*desc_2, i);
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
	vector<KeyPoint> *keypoints_1, *keypoints_2;
	Mat *desc_1, *desc_2;

	keypoints_1 = &img_keypoints[first];
	desc_1 = &img_descriptors[first];

	keypoints_2 = &img_keypoints[second];
	desc_2 = &img_descriptors[second];

	calc_dist_mat(*desc_1, *desc_2, true);

	int feature_count_1 = keypoints_1->size();
	int feature_count_2 = keypoints_2->size();

	signature_tt<double> signature_1;
	signature_tt<double> signature_2;

	signature_1.n = feature_count_1;
	signature_2.n = feature_count_2;

	signature_1.Features = new feature_tt[feature_count_1];
	signature_2.Features = new feature_tt[feature_count_2];

	signature_1.Weights = new double[feature_count_1];
	signature_2.Weights = new double[feature_count_2];

	for (int i = 0; i < feature_count_1; ++i) {
		signature_1.Weights[i] = calc_scalar_manitude(*desc_1, i);
		signature_1.Features[i] = i;
	}

	for (int i = 0; i < feature_count_2; ++i) {
		signature_2.Weights[i] = calc_scalar_manitude(*desc_2, i);
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
