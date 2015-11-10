#!/usr/bin/python

import re
import sys
import operator
import math

def main():
	args = sys.argv[1:]
	if len(args) < 2:
		print 'Usage: query.py <emd_thresh> <results_file>'
		sys.exit(1)

	th = float(args[0])
	filepath = args[1]
	results = {}
	max_dist = 0.0
	min_dist = 9999999999.0
	lines = []

	file = open(filepath, "r");
	for row in file:
		columns = row.split()
		img1 = find_filename(columns[0])
		img2 = find_filename(columns[1])
		dist = float(columns[2])
		if dist > max_dist:
			max_dist = dist
		if dist < min_dist:
			min_dist = dist
		lines.append((img1, img2, dist))

	for line in lines:
		img1 = line[0]
		img2 = line[1]
		dist = line[2]
		norm = (dist - min_dist) / (max_dist - min_dist) * 100
		if norm < th:
			update_results(results, img1, img2, norm)
		#print "%-5s %-5s %.2f" % (img1, img2, norm)

	total_img_count = len(results)
	group_counts = [0]*total_img_count
	for img1 in results:
		group1 = int(find_group(img1))
		group_counts[group1] += 1
		results[img1] = sorted(results[img1], key=lambda x: x[1])

	results = sorted(results.items(), key=path_key)
	
	variation_errors = [0] * 51
	total_recall = 0.0
	total_precision = 0.0
	for img1, imgs in results:
		true_count = 0.0
		false_negatives = []
		group1 = int(find_group(img1))
		for img2, dist in imgs:
			group2 = int(find_group(img2))
			if group1 == group2:
				true_count += 1
			else:
				variation_errors[int(find_variation(img2))] += 1
				false_negatives.append(img2)
		
		recall = true_count / group_counts[group1] * 100
		precision = true_count / len(imgs) * 100
		total_recall += recall
		total_precision += precision

		#print img1 + ':', '(recall: ' + str(recall) + ')', '(precision: ' + str(precision) + ')',
		#print false_negatives,
		#print

	#print 'Variation errors:'
	#for i in range(len(variation_errors)):
	#	print i, variation_errors[i]
	print 'Total image count:', total_img_count
	print 'Mean recall:', total_recall / total_img_count
	print 'Mean precision:', total_precision / total_img_count

def find_max_group(list):
	dict = {}
	for img2, dist in list:
		group2 = find_group(img2)
		if not dict.get(group2):
			dict[group2] = 0
		dict[group2] += 1
	max_group = max(dict.iteritems(), key=operator.itemgetter(1))[0]
	return max_group

def is_in_same_group(img1, img2):
	group1 = find_group(img1)
	group2 = find_group(img2)
	return group1 == group2

def find_filename(path):
	regex = "^.*/(.*)\..*$"
	return re.findall(regex, path)[0]

def find_group(path):
	regex = "^([\d]+)_.*$"
	return re.findall(regex, path)[0]

def find_variation(path):
	regex = "^.*_([\d]+)$"
	return re.findall(regex, path)[0]

def update_results(results, img1, img2, dist, rec=True):
	if not results.get(img1):
		results[img1] = []
	
	results[img1].append((img2, dist))

	if rec:
		update_results(results, img2, img1, dist, False)

def path_key(tuple):
	img1 = tuple[0]
	regex = "^([\d]+)_r?([\d]+)$"
	found = re.findall(regex, img1)
	return int(found[0][0] + found[0][1])

if __name__ == '__main__':
	main()
