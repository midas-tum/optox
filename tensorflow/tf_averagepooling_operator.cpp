///@file tf_averagepooling_operator.cpp
///@brief tensorflow wrappers for the averagepooling operator
///@author Kaijie Mo <mokaijie5@gmail.com> 
///@date 05.2021

#include <iostream>
#include <vector>
#include "tf_utils.h"
#include "operators/averagepooling_operator.h"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "tensorflow/core/framework/register_types.h"
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using namespace tensorflow;
using namespace std;


int getWindowedOutputSize(int input_size, int filter_size,
    int dilation_rate,
    int stride, int pad, int ceil_mode,
    std::string padding_mode)

{
    int output_size = 0;

    if (stride <= 0)
    {
        return (0);
    }
    if (dilation_rate < 1)
    {
        return (0);
    }
    int effective_filter_size = (filter_size - 1) *dilation_rate + 1;
    if(padding_mode.compare("VALID")==0){

	int size_out = (input_size + 2 * pad - effective_filter_size + stride) / stride;
	return size_out;
            }

    else if(padding_mode.compare("SAME")==0){
            /*SAME */
		int  size_out= ceil( input_size*1.0 / stride);
		return size_out;
            
    }
    return 0;
}

/**
 *register the operation with necessary options
 */

REGISTER_OP("Averagepooling1d")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Output("output_real: T")
    .Output("output_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

REGISTER_OP("Averagepooling1dGradBackward")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Input("output_real: T")
    .Input("output_imag: T")
    .Input("top_diff_real: T")
    .Input("top_diff_imag: T")
    .Output("bottom_diff_real: T")
    .Output("bottom_diff_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

template <typename T>
class TFAveragePooling1dOperator : public OpKernel {
 public:
  explicit TFAveragePooling1dOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    TensorShape input_shape = x_tensor1.shape();
    int batch = 0;
    int height_in_ = 0;
    int channels_ = 0;

    if (input_shape.dims() == 2) {
      height_in_ = input_shape.dim_size(0);
      channels_ = input_shape.dim_size(1);
    }

    else if (input_shape.dims() == 3) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      channels_ = input_shape.dim_size(2);
    } else {
    }
    int height_out_ = getWindowedOutputSize(
        height_in_, kernel_h_, dilation_rate_h_, stride_h_, pad_h_, ceil_mode_, padding_mode_);

    TensorShape output_shape = input_shape;
    if (input_shape.dims() == 2) {
      output_shape.set_dim(0, height_out_);
    } else if (input_shape.dims() == 3) {
      output_shape.set_dim(1, height_out_);
    } else {
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor2));

    /*compute the output */

    if (input_shape.dims() == 2) {
      auto input_real = getDTensorTensorflow<T, 2>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 2>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 2>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 2>(*output_tensor2);
      optox::AveragePooling1d_Operator<T> op(
          height_in_, kernel_h_, stride_h_, channels_, alpha_, beta_, pad_h_,
          dilation_rate_h_, batch, ceil_mode_,padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    } else if (input_shape.dims() == 3) {
      auto input_real = getDTensorTensorflow<T, 3>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 3>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 3>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 3>(*output_tensor2);
      optox::AveragePooling1d_Operator<T> op(
          height_in_, kernel_h_, stride_h_, channels_, alpha_, beta_, pad_h_,
          dilation_rate_h_, batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int stride_h_;
  int pad_h_;
  int channels_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling1d")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling1dOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU  

template <typename T>
class TFAveragePooling1dBackwardOperator : public OpKernel {
 public:
  explicit TFAveragePooling1dBackwardOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
 OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */

    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    const Tensor &x_tensor3 = context->input(2);
    const Tensor &x_tensor4 = context->input(3);
    const Tensor &x_tensor5 = context->input(4);
    const Tensor &x_tensor6 = context->input(5);
    TensorShape input_shape = x_tensor1.shape();

    int batch = 0;
    int height_in_ = 0;
    int channels_ = 0;

    if (input_shape.dims() == 2) {
      height_in_ = input_shape.dim_size(0);
      channels_ = input_shape.dim_size(1);
    } else if (input_shape.dims() == 3) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      channels_ = input_shape.dim_size(2);
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, input_shape, &output_tensor2));

    /*compute the output */
    if (input_shape.dims() == 2) {
      auto input_real = getDTensorTensorflow<T, 2>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 2>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 2>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 2>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 2>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 2>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 2>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 2>(x_tensor6);

      optox::AveragePooling1d_Operator<T> op(
          height_in_, kernel_h_, stride_h_, channels_, alpha_, beta_, pad_h_,
          dilation_rate_h_, batch, ceil_mode_,padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    } else if (input_shape.dims() == 3) {
      auto input_real = getDTensorTensorflow<T, 3>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 3>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 3>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 3>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 3>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 3>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 3>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 3>(x_tensor6);

      optox::AveragePooling1d_Operator<T> op(
          height_in_, kernel_h_, stride_h_, channels_, alpha_, beta_, pad_h_,
          dilation_rate_h_, batch, ceil_mode_,padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int stride_h_;
  int pad_h_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling1dGradBackward")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling1dBackwardOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU  

REGISTER_OP("Averagepooling2d")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Output("output_real: T")
    .Output("output_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

REGISTER_OP("Averagepooling2dGradBackward")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Input("output_real: T")
    .Input("output_imag: T")
    .Input("top_diff_real: T")
    .Input("top_diff_imag: T")
    .Output("bottom_diff_real: T")
    .Output("bottom_diff_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");


template <typename T>
class TFAveragePooling2dOperator : public OpKernel {
 public:
  explicit TFAveragePooling2dOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    TensorShape input_shape = x_tensor1.shape();
    int batch = 0;
    int height_in_ = 0;
    int width_in_ = 0;
    int channels_ = 0;
    if (input_shape.dims() == 3) {
      height_in_ = input_shape.dim_size(0);
      width_in_ = input_shape.dim_size(1);
      channels_ = input_shape.dim_size(2);
    }

    else if (input_shape.dims() == 4) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      width_in_ = input_shape.dim_size(2);
      channels_ = input_shape.dim_size(3);
    } else {
    }

    int height_out_ = getWindowedOutputSize(
        height_in_, kernel_h_, dilation_rate_h_, stride_h_, pad_h_, ceil_mode_, padding_mode_);
    int width_out_ = getWindowedOutputSize(
        width_in_, kernel_w_, dilation_rate_w_, stride_w_, pad_w_, ceil_mode_,  padding_mode_);

    TensorShape output_shape = input_shape;
    if (input_shape.dims() == 3) {
      output_shape.set_dim(0, height_out_);
      output_shape.set_dim(1, width_out_);
    } else if (input_shape.dims() == 4) {
      output_shape.set_dim(1, height_out_);
      output_shape.set_dim(2, width_out_);
    } else {
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor2));

    /*compute the output */

    if (input_shape.dims() == 3) {
      auto input_real = getDTensorTensorflow<T, 3>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 3>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 3>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 3>(*output_tensor2);
      optox::AveragePooling2d_Operator<T> op(
          height_in_, width_in_, kernel_h_, kernel_w_, stride_h_, stride_w_,
          channels_, alpha_, beta_, pad_h_, pad_w_, dilation_rate_h_,
          dilation_rate_w_, batch, ceil_mode_,  padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    } else if (input_shape.dims() == 4) {
      auto input_real = getDTensorTensorflow<T, 4>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 4>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 4>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 4>(*output_tensor2);
      optox::AveragePooling2d_Operator<T> op(
          height_in_, width_in_, kernel_h_, kernel_w_, stride_h_, stride_w_,
          channels_, alpha_, beta_, pad_h_, pad_w_, dilation_rate_h_,
          dilation_rate_w_, batch, ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int channels_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling2d")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling2dOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
class TFAveragePooling2dBackwardOperator : public OpKernel {
 public:
  explicit TFAveragePooling2dBackwardOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */

    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    const Tensor &x_tensor3 = context->input(2);
    const Tensor &x_tensor4 = context->input(3);
    const Tensor &x_tensor5 = context->input(4);
    const Tensor &x_tensor6 = context->input(5);
    TensorShape input_shape = x_tensor1.shape();

    int batch = 0;
    int height_in_ = 0;
    int width_in_ = 0;
    int channels_ = 0;

    if (input_shape.dims() == 3) {
      height_in_ = input_shape.dim_size(0);
      width_in_ = input_shape.dim_size(1);
      channels_ = input_shape.dim_size(2);
    } else if (input_shape.dims() == 4) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      width_in_ = input_shape.dim_size(2);
      channels_ = input_shape.dim_size(3);
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, input_shape, &output_tensor2));

    /*compute the output */
    if (input_shape.dims() == 3) {
      auto input_real = getDTensorTensorflow<T, 3>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 3>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 3>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 3>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 3>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 3>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 3>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 3>(x_tensor6);

      optox::AveragePooling2d_Operator<T> op(
          height_in_, width_in_, kernel_h_, kernel_w_, stride_h_, stride_w_,
          channels_, alpha_, beta_, pad_h_, pad_w_, dilation_rate_h_,
          dilation_rate_w_, batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    } else if (input_shape.dims() == 4) {
      auto input_real = getDTensorTensorflow<T, 4>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 4>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 4>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 4>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 4>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 4>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 4>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 4>(x_tensor6);

      optox::AveragePooling2d_Operator<T> op(
          height_in_, width_in_, kernel_h_, kernel_w_, stride_h_, stride_w_,
          channels_, alpha_, beta_, pad_h_, pad_w_, dilation_rate_h_,
          dilation_rate_w_, batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling2dGradBackward")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling2dBackwardOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

/*---------------- AveragePooling3d ---------------- */

/**
 *register the operation with necessary options
 */
REGISTER_OP("Averagepooling3d")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Output("output_real: T")
    .Output("output_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("kernel_d: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("stride_d: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("pad_d: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("dilation_rate_d: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

REGISTER_OP("Averagepooling3dGradBackward")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Input("output_real: T")
    .Input("output_imag: T")
    .Input("top_diff_real: T")
    .Input("top_diff_imag: T")
    .Output("bottom_diff_real: T")
    .Output("bottom_diff_imag: T")
    .Attr("T: {float32, float64}")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("kernel_d: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("stride_d: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("pad_d: int >= 0")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("dilation_rate_d: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

template <typename T>
class TFAveragePooling3dOperator : public OpKernel {
 public:
  explicit TFAveragePooling3dOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_d", &kernel_d_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_d", &stride_d_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_d", &pad_d_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_d", &dilation_rate_d_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    TensorShape input_shape = x_tensor1.shape();

    int batch = 0;
    int height_in_ = 0;
    int width_in_ = 0;
    int depth_in_ = 0;
    int channels_ = 0;

    if (input_shape.dims() == 4) {
      height_in_ = input_shape.dim_size(0);
      width_in_ = input_shape.dim_size(1);
      depth_in_ = input_shape.dim_size(2);
      channels_ = input_shape.dim_size(3);
    } else if (input_shape.dims() == 5) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      width_in_ = input_shape.dim_size(2);
      depth_in_ = input_shape.dim_size(3);
      channels_ = input_shape.dim_size(4);
    } else {
    }

    int height_out_ = getWindowedOutputSize(
        height_in_, kernel_h_, dilation_rate_h_, stride_h_, pad_h_, ceil_mode_,  padding_mode_);
    int width_out_ = getWindowedOutputSize(
        width_in_, kernel_w_, dilation_rate_w_, stride_w_, pad_w_, ceil_mode_, padding_mode_);
    int depth_out_ = getWindowedOutputSize(
        depth_in_, kernel_w_, dilation_rate_d_, stride_w_, pad_d_, ceil_mode_,  padding_mode_);

    TensorShape output_shape = input_shape;

    if (input_shape.dims() == 4) {
      output_shape.set_dim(0, height_out_);
      output_shape.set_dim(1, width_out_);
      output_shape.set_dim(2, depth_out_);
    } else if (input_shape.dims() == 5) {
      output_shape.set_dim(1, height_out_);
      output_shape.set_dim(2, width_out_);
      output_shape.set_dim(3, depth_out_);
    } else {
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor2));

    /*compute the output */
    if (input_shape.dims() == 4) {
      auto input_real = getDTensorTensorflow<T, 4>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 4>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 4>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 4>(*output_tensor2);
      optox::AveragePooling3d_Operator<T> op(
          height_in_, width_in_, depth_in_, kernel_h_, kernel_w_, kernel_d_,
          stride_h_, stride_w_, stride_d_, channels_, alpha_, beta_, pad_h_,
          pad_w_, pad_d_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
          batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    } else if (input_shape.dims() == 5) {
      auto input_real = getDTensorTensorflow<T, 5>(x_tensor1);
      auto output_real = getDTensorTensorflow<T, 5>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 5>(x_tensor2);
      auto output_imag = getDTensorTensorflow<T, 5>(*output_tensor2);
      optox::AveragePooling3d_Operator<T> op(
          height_in_, width_in_, depth_in_, kernel_h_, kernel_w_, kernel_d_,
          stride_h_, stride_w_, stride_d_, channels_, alpha_, beta_, pad_h_,
          pad_w_, pad_d_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
          batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.forward({output_real.get(), output_imag.get()},
                 {input_real.get(), input_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int kernel_w_;
  int kernel_d_;
  int stride_h_;
  int stride_w_;
  int stride_d_;
  int pad_h_;
  int pad_w_;
  int pad_d_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  int dilation_rate_d_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling3d")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling3dOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
class TFAveragePooling3dBackwardOperator : public OpKernel {
 public:
  explicit TFAveragePooling3dBackwardOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_d", &kernel_d_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_d", &stride_d_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_d", &pad_d_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_d", &dilation_rate_d_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    const Tensor &x_tensor3 = context->input(2);
    const Tensor &x_tensor4 = context->input(3);
    const Tensor &x_tensor5 = context->input(4);
    const Tensor &x_tensor6 = context->input(5);
    TensorShape input_shape = x_tensor1.shape();

    int batch = 0;
    int height_in_ = 0;
    int width_in_ = 0;
    int depth_in_ = 0;
    int channels_ = 0;
    if (input_shape.dims() == 4) {
      height_in_ = input_shape.dim_size(0);
      width_in_ = input_shape.dim_size(1);
      depth_in_ = input_shape.dim_size(2);
      channels_ = input_shape.dim_size(3);
    } else if (input_shape.dims() == 5) {
      batch = input_shape.dim_size(0);
      height_in_ = input_shape.dim_size(1);
      width_in_ = input_shape.dim_size(2);
      depth_in_ = input_shape.dim_size(3);
      channels_ = input_shape.dim_size(4);
    } else {
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, input_shape, &output_tensor2));

    /*compute the output */
    if (input_shape.dims() == 4) {
      auto input_real = getDTensorTensorflow<T, 4>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 4>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 4>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 4>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 4>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 4>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 4>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 4>(x_tensor6);
      optox::AveragePooling3d_Operator<T> op(
          height_in_, width_in_, depth_in_, kernel_h_, kernel_w_, kernel_d_,
          stride_h_, stride_w_, stride_d_, channels_, alpha_, beta_, pad_h_,
          pad_w_, pad_d_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
          batch,ceil_mode_, padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    } else if (input_shape.dims() == 5) {
      auto input_real = getDTensorTensorflow<T, 5>(x_tensor1);
      auto bottom_diff_real = getDTensorTensorflow<T, 5>(*output_tensor1);
      auto input_imag = getDTensorTensorflow<T, 5>(x_tensor2);
      auto bottom_diff_imag = getDTensorTensorflow<T, 5>(*output_tensor2);
      auto output_real = getDTensorTensorflow<T, 5>(x_tensor3);
      auto output_imag = getDTensorTensorflow<T, 5>(x_tensor4);
      auto top_diff_real = getDTensorTensorflow<T, 5>(x_tensor5);
      auto top_diff_imag = getDTensorTensorflow<T, 5>(x_tensor6);
      optox::AveragePooling3d_Operator<T> op(
          height_in_, width_in_, depth_in_, kernel_h_, kernel_w_, kernel_d_,
          stride_h_, stride_w_, stride_d_, channels_, alpha_, beta_, pad_h_,
          pad_w_, pad_d_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
          batch, ceil_mode_,padding_mode_);

      op.setStream(context->eigen_device<GPUDevice>().stream());
      op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
                 {input_real.get(), input_imag.get(), output_real.get(),
                  output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
    }
  }

 private:
  int kernel_h_;
  int kernel_w_;
  int kernel_d_;
  int stride_h_;
  int stride_w_;
  int stride_d_;
  int pad_h_;
  int pad_w_;
  int pad_d_;
  float alpha_;
  float beta_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  int dilation_rate_d_;
  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling3dGradBackward")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling3dBackwardOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU   

REGISTER_OP("Averagepooling4d")

    .Input("input_real: T")
    .Input("input_imag: T")
    .Output("output_real: T")
    .Output("output_imag: T")

    .Attr("T: {float32, float64}")

    .Attr("kernel_t: int >= 0")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("kernel_d: int >= 0")

    .Attr("stride_t: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("stride_d: int >= 0")

    .Attr("pad_t: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("pad_d: int >= 0")

    .Attr("alpha: float")
    .Attr("beta: float")

    .Attr("dilation_rate_t: int >= 1")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("dilation_rate_d: int >= 1")

    .Attr("batch: int >= 0")
    .Attr("time_in: int >= 1")
    .Attr("height_in: int >= 1")
    .Attr("width_in: int >= 1")
    .Attr("depth_in: int >= 1")
    .Attr("channels: int >= 1")

    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");

REGISTER_OP("Averagepooling4dGradBackward")
    .Input("input_real: T")
    .Input("input_imag: T")
    .Input("output_real: T")
    .Input("output_imag: T")
    .Input("top_diff_real: T")
    .Input("top_diff_imag: T")
    .Output("bottom_diff_real: T")
    .Output("bottom_diff_imag: T")
    .Attr("T: {float32, float64}")

    .Attr("kernel_t: int >= 0")
    .Attr("kernel_h: int >= 0")
    .Attr("kernel_w: int >= 0")
    .Attr("kernel_d: int >= 0")

    .Attr("stride_t: int >= 0")
    .Attr("stride_h: int >= 0")
    .Attr("stride_w: int >= 0")
    .Attr("stride_d: int >= 0")

    .Attr("pad_t: int >= 0")
    .Attr("pad_h: int >= 0")
    .Attr("pad_w: int >= 0")
    .Attr("pad_d: int >= 0")

    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("batch: int >= 0")
    .Attr("time_in: int >= 1")
    .Attr("height_in: int >= 1")
    .Attr("width_in: int >= 1")
    .Attr("depth_in: int >= 1")
    .Attr("channels: int >= 1")

    .Attr("dilation_rate_t: int >= 1")
    .Attr("dilation_rate_h: int >= 1")
    .Attr("dilation_rate_w: int >= 1")
    .Attr("dilation_rate_d: int >= 1")
    .Attr("mode: {'VALID','SAME'}")
    .Attr("ceil_mode: int=0");
template <typename T>
class TFAveragePooling4dOperator : public OpKernel {
 public:
  explicit TFAveragePooling4dOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_t", &kernel_t_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_d", &kernel_d_));

    OP_REQUIRES_OK(context, context->GetAttr("stride_t", &stride_t_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_d", &stride_d_));

    OP_REQUIRES_OK(context, context->GetAttr("pad_t", &pad_t_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_d", &pad_d_));

    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));

    OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
    OP_REQUIRES_OK(context, context->GetAttr("time_in", &time_in_));
    OP_REQUIRES_OK(context, context->GetAttr("height_in", &height_in_));
    OP_REQUIRES_OK(context, context->GetAttr("width_in", &width_in_));
    OP_REQUIRES_OK(context, context->GetAttr("depth_in", &depth_in_));
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));

    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_t", &dilation_rate_t_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_d", &dilation_rate_d_));

    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    TensorShape input_shape = x_tensor1.shape();

    int time_out_ = getWindowedOutputSize(time_in_, kernel_t_, dilation_rate_t_,
                                          stride_t_, pad_t_, ceil_mode_, padding_mode_);
    int height_out_ = getWindowedOutputSize(
        height_in_, kernel_h_, dilation_rate_h_, stride_h_, pad_h_, ceil_mode_,  padding_mode_);
    int width_out_ = getWindowedOutputSize(
        width_in_, kernel_w_, dilation_rate_w_, stride_w_, pad_w_, ceil_mode_,  padding_mode_);
    int depth_out_ = getWindowedOutputSize(
        depth_in_, kernel_w_, dilation_rate_d_, stride_w_, pad_d_, ceil_mode_, padding_mode_);

    TensorShape output_shape = input_shape;

    if (batch_ != 0) {
      output_shape.set_dim(
          0, time_out_ * height_out_ * width_out_ * depth_out_ * channels_);
    } else if (batch_ > 0) {
      output_shape.set_dim(0, batch_ * time_out_ * height_out_ * width_out_ *
                                  depth_out_ * channels_);
    } else {
    }

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor2));

    /*compute the output */
    auto input_real = getDTensorTensorflow<T, 1>(x_tensor1);
    auto output_real = getDTensorTensorflow<T, 1>(*output_tensor1);
    auto input_imag = getDTensorTensorflow<T, 1>(x_tensor2);
    auto output_imag = getDTensorTensorflow<T, 1>(*output_tensor2);

    optox::AveragePooling4d_Operator<T> op(
        time_in_, height_in_, width_in_, depth_in_, kernel_t_, kernel_h_,
        kernel_w_, kernel_d_, stride_t_, stride_h_, stride_w_, stride_d_,
        channels_, alpha_, beta_, pad_t_, pad_h_, pad_w_, pad_d_,
        dilation_rate_t_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
        batch_,ceil_mode_, padding_mode_);

    op.setStream(context->eigen_device<GPUDevice>().stream());
    op.forward({output_real.get(), output_imag.get()},
               {input_real.get(), input_imag.get()});
  }

 private:
  int kernel_t_;
  int kernel_h_;
  int kernel_w_;
  int kernel_d_;

  int stride_t_;
  int stride_h_;
  int stride_w_;
  int stride_d_;

  int pad_t_;
  int pad_h_;
  int pad_w_;
  int pad_d_;

  float alpha_;
  float beta_;

  int dilation_rate_t_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  int dilation_rate_d_;

  int batch_;
  int time_in_;
  int height_in_;
  int width_in_;
  int depth_in_;
  int channels_;

  std::string padding_mode_;
  int ceil_mode_;
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling4d")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling4dOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
class TFAveragePooling4dGradBackwardOperator : public OpKernel {
 public:
  explicit TFAveragePooling4dGradBackwardOperator(OpKernelConstruction *context)
      : OpKernel(context) { /*Get attributes */
    OP_REQUIRES_OK(context, context->GetAttr("kernel_t", &kernel_t_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_d", &kernel_d_));

    OP_REQUIRES_OK(context, context->GetAttr("stride_t", &stride_t_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_d", &stride_d_));

    OP_REQUIRES_OK(context, context->GetAttr("pad_t", &pad_t_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_d", &pad_d_));

    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));

    OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
    OP_REQUIRES_OK(context, context->GetAttr("time_in", &time_in_));
    OP_REQUIRES_OK(context, context->GetAttr("height_in", &height_in_));
    OP_REQUIRES_OK(context, context->GetAttr("width_in", &width_in_));
    OP_REQUIRES_OK(context, context->GetAttr("depth_in", &depth_in_));
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));

    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_t", &dilation_rate_t_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_h", &dilation_rate_h_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_w", &dilation_rate_w_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dilation_rate_d", &dilation_rate_d_));

    OP_REQUIRES_OK(context, context->GetAttr("mode", &padding_mode_));
     OP_REQUIRES_OK(context, context->GetAttr("ceil_mode", &ceil_mode_));
  }

  void Compute(OpKernelContext *context) override { /*Grab the input tensor */
    const Tensor &x_tensor1 = context->input(0);
    const Tensor &x_tensor2 = context->input(1);
    const Tensor &x_tensor3 = context->input(2);
    const Tensor &x_tensor4 = context->input(3);
    const Tensor &x_tensor5 = context->input(4);
    const Tensor &x_tensor6 = context->input(5);
    TensorShape input_shape = x_tensor1.shape();

    /*allocate the output */
    Tensor *output_tensor1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &output_tensor1));
    Tensor *output_tensor2 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, input_shape, &output_tensor2));

    /*compute the output */

    auto input_real = getDTensorTensorflow<T, 1>(x_tensor1);
    auto bottom_diff_real = getDTensorTensorflow<T, 1>(*output_tensor1);
    auto input_imag = getDTensorTensorflow<T, 1>(x_tensor2);
    auto bottom_diff_imag = getDTensorTensorflow<T, 1>(*output_tensor2);
    auto output_real = getDTensorTensorflow<T, 1>(x_tensor3);
    auto output_imag = getDTensorTensorflow<T, 1>(x_tensor4);
    auto top_diff_real = getDTensorTensorflow<T, 1>(x_tensor5);
    auto top_diff_imag = getDTensorTensorflow<T, 1>(x_tensor6);
    optox::AveragePooling4d_Operator<T> op(
        time_in_, height_in_, width_in_, depth_in_, kernel_t_, kernel_h_,
        kernel_w_, kernel_d_, stride_t_, stride_h_, stride_w_, stride_d_,
        channels_, alpha_, beta_, pad_t_, pad_h_, pad_w_, pad_d_,
        dilation_rate_t_, dilation_rate_h_, dilation_rate_w_, dilation_rate_d_,
        batch_,ceil_mode_, padding_mode_);

    op.setStream(context->eigen_device<GPUDevice>().stream());
    op.adjoint({bottom_diff_real.get(), bottom_diff_imag.get()},
               {input_real.get(), input_imag.get(), output_real.get(),
                output_imag.get(), top_diff_real.get(), top_diff_imag.get()});
  }

 private:
  int kernel_t_;
  int kernel_h_;
  int kernel_w_;
  int kernel_d_;

  int stride_t_;
  int stride_h_;
  int stride_w_;
  int stride_d_;

  int pad_t_;
  int pad_h_;
  int pad_w_;
  int pad_d_;
  float alpha_;
  float beta_;

  int batch_;
  int time_in_;
  int height_in_;
  int width_in_;
  int depth_in_;
  int channels_;

  int dilation_rate_t_;
  int dilation_rate_h_;
  int dilation_rate_w_;
  int dilation_rate_d_;
  std::string padding_mode_;
  int ceil_mode_;
  
};

#define REGISTER_GPU(type)\
REGISTER_KERNEL_BUILDER(\
    Name("Averagepooling4dGradBackward")\
    .Device(DEVICE_GPU)\
    .TypeConstraint<type> ("T"), \
    TFAveragePooling4dGradBackwardOperator < type>)\

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU  
    
