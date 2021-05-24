#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "SimilarityCalculator.h"

using namespace std;

int main()
{
    int similarity_thd;
    string str, dummy;
    vector<string> input_paths;

    // read similarity treshold
    cin >> similarity_thd;
    cin.get(); // removing '\n' from cin

    // read paths of image files
    while (getline(cin, str) && str.length() > 0) {
        input_paths.emplace_back(str);
    } 
  
    // try to read files, make pairs <filename, cv_image>
    vector< pair<string, cv::Mat> > images;
    for (const auto& img_path : input_paths) {
        auto img = cv::imread(img_path);
        if (img.empty()) {
            cout << "File " << img_path << " not found!" << endl;
            continue;
        }
        images.emplace_back(make_pair(img_path, img));
    }

    auto sim_calc = SimilarityCalculator();

    // calculate similarity of images
    for (auto it = images.cbegin(); it < images.cend() - 1; ++it)
        for (auto jt = it + 1; jt < images.cend(); ++jt) {
            auto similarity_score = int(sim_calc.calculate_similarity(it->second, jt->second) * 100);

            // print file names with enough similarity
            if (similarity_score >= similarity_thd) cout << it->first << ", " << jt->first << ", " << similarity_score << '%' << endl;
        }

    cin.get();
    return 0;
}
