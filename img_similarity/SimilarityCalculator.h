#pragma once

#include <opencv2/opencv.hpp>

class SimilarityCalculator
{
public:
	// Ctor
	explicit SimilarityCalculator(double weight_desñriptor_distance = 0.5, double weight_hist_correlation = 0.5)
		:weight_des_distance(weight_desñriptor_distance), weight_hist_corr(weight_hist_correlation)
	{

	}
	// Dtor
	virtual ~SimilarityCalculator(){}

private:
	// weights
	double weight_des_distance;
	double weight_hist_corr;

	std::vector<cv::Mat> get_bgr_hists(const cv::Mat& img);

protected:
	// functions for calculating similarity
	virtual double hist_corellation(const cv::Mat& img1, const cv::Mat& img2);
	virtual double descriptors_distance(const cv::Mat& img1, const cv::Mat& img2);

public:
	// function for getting similarity score
	virtual double calculate_similarity(const cv::Mat& img1, const cv::Mat& img2);
};
