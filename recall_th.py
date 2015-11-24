#!/usr/bin/python

import re
import sys
import operator
import math

def main():
	args = sys.argv[1:]
	if len(args) < 1:
		print 'Usage: recall.py <results_file> <original>'
		sys.exit(1)

	filepath = args[0]
	only_org = bool(args[1]) if len(args) == 2 else False
	results = {}
	max_dist = 0
	min_dist = float('inf')
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
		update_results(results, img1, img2, norm, False)
		#print "%-5s %-5s %.2f" % (img1, img2, norm)
		
	# count variations and sort by dist
	group_counts = [0]*1000
	var_counts = [0]*51
	for img1 in results:
		group1 = int(find_group(img1))
		var1 = int(find_variation(img1))
		group_counts[group1] += 1
		var_counts[var1] += 1
		results[img1] = sorted(results[img1], cmp=dist_cmp)

	results = sorted(results.items(), key=path_key)

	# for img1, imgs in results:
	# 	print img1
	# 	for img2, dist, img1 in imgs:
	# 		print "\t", img2, dist
	
	total_img_count = len(results)
	values = {}
	precision_sum = [0]*101
	counts = [0]*101
	recall_sum = [0]*101
	variation_errors = [0] * 51
	total_precision = 0.0
	total_recall = 0.0
	for img1, imgs in results:
		if only_org:
			var1 = int(find_variation(img1))
			if var1 != 0:
				continue
		true_count = 0.0
		false_count = 0.0
		img_count = 0
		false_negatives = []
		img_values = []
		group1 = int(find_group(img1))
		last_dist = 0;
		last_recall = 0;
		last_precision = 100;
		for img2, dist, img1 in imgs:
			group2 = int(find_group(img2))
			if group1 == group2:
				true_count += 1
			else:
				false_count += 1
				variation_errors[int(find_variation(img2))] += 1
				false_negatives.append(img2)
			img_count += 1
			recall = true_count / (group_counts[group1]-1) * 100
			precision = true_count / img_count * 100
			# if (recall >= recall_th):
			# 	total_recall += recall
			# 	total_precision += precision
			# 	break
			for i in range(int(last_dist), int(dist)+1):
				recall_sum[i] += last_recall
				precision_sum[i] += last_precision
				counts[i] += 1
			last_dist = dist+1
			last_recall = recall
			last_precision = precision
			# img_values.append((recall, precision, dist))
			# print "{:>10f} {:>10f} {:>10f}".format(recall, precision, dist)
		for i in range(int(last_dist), 101):
			recall_sum[i] += last_recall
			precision_sum[i] += last_precision
			counts[i] += 1
		# print '-' * 30
		# for i in range(len(recall_sum)):
		# 	print "{:>4} {:>10f} {:>10f} {:>10f}".format(i, recall_sum[i], precision_sum[i], counts[i])
		# values[img1] = img_values
		# print img1 + ':', '(recall: ' + str(recall) + ')', '(precision: ' + str(precision) + ')',
		# print false_negatives,
		# print

	print 'Image count:', total_img_count
	print '-' * 40
	for i in range(len(recall_sum)):
		print recall_sum[i] / counts[i], precision_sum[i] / counts[i]
	#print 'Variation errors:'
	#for i in range(len(variation_errors)):
	#	print i, variation_errors[i]
	# print 'Mean recall:', total_recall / total_img_count
	# print 'Mean precision:', total_precision / total_img_count

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
	
	results[img1].append((img2, dist, img1))

	if rec:
		update_results(results, img2, img1, dist, False)

def path_key(tuple):
	img1 = tuple[0]
	return name_to_int(img1)

def name_to_int(name):
	regex = "^([\d]+)_r?([\d]+)$"
	found = re.findall(regex, name)
	var_num = int(found[0][1])
	obj_num = int(found[0][0])
	return obj_num * 100 + var_num

def dist_cmp(x, y):
	if x[1] == y[1]:
		gm = find_group(x[2])
		gx = find_group(x[0])
		gy = find_group(y[0])
		if gx == gy:
			return 0;
		elif gm == gx:
			return -1
		elif gm == gy:
			return 1
		else:
			return 0
	return -1 if x[1] < y[1] else 1

if __name__ == '__main__':
	main()
