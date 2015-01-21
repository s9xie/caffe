#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageContextDataLayer<Dtype>::~ImageContextDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageContextDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int upsampling_rate = this->layer_param_.image_data_param().upsampling_rate();
  //const int -> int must change this
  int new_height = this->layer_param_.image_data_param().new_height();
  int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string img_filename;
  string gt_filename;
  while (infile >> img_filename >> gt_filename) {
    lines_.push_back(std::make_pair(img_filename, gt_filename));
  }

  // shuffle the lines, so the pair of img/gt won't change
  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }

  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  //TO-DO: make this a parameter
  int upsampling_factor = 2; //hard coded for now
  int num_cls = 21;

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
  cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    0, 0, 0);

  //Debug
  std::cout << "cv_img" << " " << cv_img << std::endl << std::endl;
  std::cout << "cv_gt"  << " " << cv_gt  << std::endl << std::endl;

  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  
  const int gt_channels = cv_gt.channels();
  const int gt_height = cv_gt.rows;
  const int gt_width = cv_gt.cols;

  CHECK((height == gt_height) && (width == gt_width)) << "GT image size should be equal to true image size";
  CHECK(gt_channels == 1) << "GT image channel number should be equal to one";

  
  new_width = width * upsampling_factor;
  new_height = height * upsampling_factor;
  //opencv resize can take src == dst without preallocation

  if (new_height > 0 && new_width > 0) {
    cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
    cv::resize(cv_gt, cv_gt, cv::Size(new_width, new_height));
  }

  // image
  //top[0] data
  //top[1] label
  //top[2] context
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);

    top[2]->Reshape(batch_size, num_cls, crop_size, crop_size);
    this->prefetch_context_.Reshape(batch_size, num_cls, crop_size, crop_size);
    this->transformed_context_.Reshape(1, num_cls, crop_size, crop_size);

  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    top[2]->Reshape(batch_size, num_cls, height, width);
    this->prefetch_context_.Reshape(batch_size, num_cls, height, width);
    this->transformed_context_.Reshape(1, num_cls, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
}

template <typename Dtype>
void ImageContextDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageContextDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  CHECK(this->prefetch_context_.count());
  CHECK(this->transformed_context_.count());

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_context = this->prefetch_context_.mutable_cpu_data();

  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  int new_height = image_data_param.new_height();
  int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    0, 0, is_color);
    cv::Mat cv_gt = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    0, 0, 0);

    //TO-DO: make this a parameter
    const int upsampling_factor = 2; //hard coded for now
    const int num_cls = 21;

    //const int channels = cv_img.channels();
    const int height = cv_img.rows;
    const int width = cv_img.cols;
    
    const int gt_channels = cv_gt.channels();
    const int gt_height = cv_gt.rows;
    const int gt_width = cv_gt.cols;
  
    CHECK((height == gt_height) && (width == gt_width)) << "GT image size should be equal to true image size";
    CHECK(gt_channels == 1) << "GT image channel number should be equal to one";
  
    
    new_width = width * upsampling_factor;
    new_height = height * upsampling_factor;

    if (new_height > 0 && new_width > 0) {
      cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
      cv::resize(cv_gt, cv_gt, cv::Size(new_width, new_height), 0, 0, cv::INTER_NEAREST);
    }

    if (!cv_img.data || !cv_gt.data) {
      continue;
    }


    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    int offset_gt = this->prefetch_context_.offset(item_id);
    CHECK(offset == offset_gt) << "fetching should be synchronized";

    this->transformed_data_.set_cpu_data(top_data + offset);
    this->transformed_context_.set_cpu_data(top_context + offset_gt);

    std::pair<int, int> hw_off = this->data_transformer_.LocTransform(cv_img, &(this->transformed_data_));
    int h_off = hw_off.first;
    int w_off = hw_off.second;

    //We want to know the "centeral" label, so we do a simple voting here for robustness.
    int crop_size = this->layer_param_.transform_param().crop_size();
    
    int max_label = 0;
    int max = 0;

    int p1 = cv_gt.at<int>(h_off + crop_size/2, w_off + crop_size/2);
    int p2 = cv_gt.at<int>(h_off + crop_size/2 - 1, w_off + crop_size/2);
    int p3 = cv_gt.at<int>(h_off + crop_size/2, w_off + crop_size/2 - 1);
    int p4 = cv_gt.at<int>(h_off + crop_size/2 - 1, w_off + crop_size/2 - 1);

    int np1 = (int)(p1 == p2) + (int)(p1 == p3) + (int)(p1 == p4);
    int np2 = (int)(p2 == p1) + (int)(p2 == p3) + (int)(p2 == p4);
    int np3 = (int)(p3 == p1) + (int)(p3 == p2) + (int)(p3 == p4);
    int np4 = (int)(p4 == p1) + (int)(p4 == p2) + (int)(p4 == p3);

    if (np1 >= np2) { max = np1; max_label = p1;} else { max = np2; max_label = p2;}
    if (np3 >= max) { max = np3; max_label = p3;}
    if (np4 >= max) max_label = p4;

    //top_label[item_id] = lines_[lines_id_].second;
    top_label[item_id] = max_label;
    // go to the next iter

    //blackout the (4) centeral pixels
    cv::Mat drop_out_mask = cv::Mat::ones(new_height, new_width, CV_8U);
    drop_out_mask.at<int>(h_off + crop_size/2, w_off + crop_size/2) = 0;
    drop_out_mask.at<int>(h_off + crop_size/2 - 1, w_off + crop_size/2) = 0;
    drop_out_mask.at<int>(h_off + crop_size/2, w_off + crop_size/2 - 1) = 0;
    drop_out_mask.at<int>(h_off + crop_size/2 - 1, w_off + crop_size/2 - 1) = 0;
    cv_gt = cv_gt.mul(drop_out_mask);

    //one-hot encoding of the gt map
    cv::Mat temp;
    vector<cv::Mat> cls_channels;
    for(int c = 0; c < num_cls; c++) {
      temp = (cv_gt == c);
      cls_channels.push_back(temp/255);
    }
    cv::Mat encoded_gt;
    cv::merge(cls_channels, encoded_gt);


    this->data_transformer_.ContextTransform(encoded_gt, &(this->transformed_context_), hw_off);

    // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    // cv::Rect roi(w_off, h_off, crop_size, crop_size);
    //x <= pt.x < x+width,<BR>y <= pt.y < y+height
    //for(int y = roi.y; y < roi.y + rect.height; y++)
       //for(int x = roi.x; x < roi.x + rect.width; x++)


    trans_time += timer.MicroSeconds();

    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageContextDataLayer);
REGISTER_LAYER_CLASS(IMAGE_CONTEXT_DATA, ImageContextDataLayer);
}  // namespace caffe
