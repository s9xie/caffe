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
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int upsampling_rate = this->layer_param_.image_data_param().upsampling_rate();
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
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
    cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
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

    top[2]->Reshape(batch_size, cls_number, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, cls_number, crop_size, crop_size);
    this->transformed_data_.Reshape(1, cls_number, crop_size, crop_size);

  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    top[2]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
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

    if (new_height > 0 && new_width > 0) {
      cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
      cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
    }

    if (!cv_img.data) {
      continue;
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_.Transform(cv_img, &(this->transformed_data_));
    
    trans_time += timer.MicroSeconds();

    top_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
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

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(IMAGE_DATA, ImageDataLayer);
}  // namespace caffe
