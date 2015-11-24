#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "FastEMD/tictoc.hpp"

using namespace cv;
using namespace std;

namespace fs = boost::filesystem;

void help();

int main(int argc, char *argv[]) {
	if (argc != 4) {
		help();
		return -1;
	}

	const char *path = argv[1];
	const char *merged_index_path = argv[2];
	int part = atoi(argv[3]);

	tictoc timer;
	timer.tic();
	
	stringstream index_path;
	string img_path;
	vector<KeyPoint> kps;
	Mat descs;
	FileStorage fs_merged(merged_index_path, FileStorage::WRITE);
	fs_merged << "index" << "[";
	for (int i = 0; i < part; ++i) {
		index_path.str("");
		index_path << path << "/index_" << i << ".yml.gz";
		cout << "Merging index from " << index_path.str() << endl;
		FileStorage fs(index_path.str(), FileStorage::READ);
		FileNode imgs = fs["index"];
		FileNodeIterator it;
		int img_count = 0;
		vector<pair<int, int> > query_ids;
		for (it = imgs.begin(); it != imgs.end(); ++it, ++img_count) {
			(*it)["path"] >> img_path;
			read((*it)["keypoints"], kps);
			(*it)["descriptors"] >> descs;

			fs_merged << "{";
			fs_merged << "path" << img_path;
			fs_merged << "keypoints" << kps;
			fs_merged << "descriptors" << descs;
			fs_merged << "}";
		}
	}
	fs_merged << "]";
	fs_merged.release();

	timer.toc();
	cout << "Time in seconds: " << timer.totalTimeSec() << endl;

	return 0;
}

void help() {
	cout << "Usage: ./merge_index <path> <merged_index_path> <part>" << endl;
}