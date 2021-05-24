#include <numeric>
#include "SimilarityCalculator.h"

using namespace std;

vector<cv::Mat> SimilarityCalculator::get_bgr_hists(const cv::Mat& img)
{
	vector<cv::Mat> bgr_plates;

	// split image by channel
	cv::split(img, bgr_plates);
	
	vector<cv::Mat> bgr_hists(bgr_plates.size());
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	// make histogramms
	for (int i = 0; i < bgr_plates.size(); i++) {
		cv::calcHist(&bgr_plates[i], 1, 0, cv::Mat(), bgr_hists[i], 1, &histSize, &histRange);
		cv::normalize(bgr_hists[i], bgr_hists[i], 0, 1, cv::NORM_MINMAX);
	}

	return bgr_hists;
}

double SimilarityCalculator::hist_corellation(const cv::Mat& img1, const cv::Mat& img2)
{
	// get b, g, r histogramms
	auto bgr_hists1 = get_bgr_hists(img1);
	auto bgr_hists2 = get_bgr_hists(img2);

	assert (bgr_hists1.size() == bgr_hists2.size());

	vector<double> channel_corr;
	// get histogramm correlation by channel
	for (int i = 0; i < bgr_hists1.size(); i++)
		channel_corr.push_back(std::abs(cv::compareHist(bgr_hists1[i], bgr_hists2[i], cv::HISTCMP_CORREL)));

	// calculate average correlation score
	return accumulate(channel_corr.cbegin(), channel_corr.cend(), decltype(channel_corr)::value_type(0)) / channel_corr.size();
}

double SimilarityCalculator::descriptors_distance(const cv::Mat& img1, const cv::Mat& img2)
{
	cv::Mat gray1, gray2;
	// get grayscale images
	cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

	// initiate ORB detector
	auto orb = cv::ORB::create();
	
	vector<cv::KeyPoint> kp1, kp2;
	cv::Mat des1, des2;

	// get keypoints and descriptors
	orb->detectAndCompute(gray1, cv::noArray(), kp1, des1);
	orb->detectAndCompute(gray2, cv::noArray(), kp2, des2);

	// initiate bruteforce matcher
	auto bf_matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
	vector< vector<cv::DMatch> > knn_matches;
	// get knn matches
	bf_matcher->knnMatch(des1, des2, knn_matches, 2);
	
	// Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.75f;
	vector<float> good_matches_distance;
	for (size_t i = 0; i < knn_matches.size(); i++)
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			good_matches_distance.push_back(knn_matches[i][0].distance);

	// normalize distance to 0...1
	cv::normalize(good_matches_distance, good_matches_distance, 1., 0., cv::NORM_INF);

	// calculate average distance
	auto score = accumulate(good_matches_distance.cbegin(),good_matches_distance.cend(),
							 decltype(good_matches_distance)::value_type(0)) / good_matches_distance.size();

	// 0. - is 100% similarity, 1. - 0%
	return 1. - score;
}

double SimilarityCalculator::calculate_similarity(const cv::Mat& img1, const cv::Mat& img2)
{
	// weighted average similarity score
	auto score = (this->weight_des_distance * descriptors_distance(img1, img2) + \
				  this->weight_hist_corr * hist_corellation(img1, img2)) / (this->weight_des_distance + this->weight_hist_corr);
	return score;
}
