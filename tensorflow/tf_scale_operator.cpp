///@file tf_scale_operator.cpp
///@brief tensorflow wrappers for the scale operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 03.2022

#include <vector>

#include "tf_utils.h"
#include "operators/scale_operator.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "tensorflow/core/framework/register_types.h"
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>

using namespace tensorflow;
using namespace std;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Operator registration
REGISTER_OP("Scale")
    .Attr("T: numbertype")
    .Input("x: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ScaleGrad")
    .Attr("T: numbertype")
    .Input("x: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename T>
class TFScaleOperator : public OpKernel {
public:
	
	explicit TFScaleOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);
		TensorShape output_shape = x_tensor.shape();

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 1>(x_tensor);
		auto output = getDTensorTensorflow<T, 1>(*output_tensor);
		
		optox::ScaleOperator<T> op;
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}
};

#define REGISTER_GPU(dtype) \
	REGISTER_KERNEL_BUILDER( \
		Name("Scale") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<typename optox::tf<dtype>::type >("T"), \
		TFScaleOperator<dtype>) \

REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(float2);
REGISTER_GPU(double2);

#undef REGISTER_GPU

template <typename T>
class TFScaleGradOperator : public OpKernel {
public:
	
	explicit TFScaleGradOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);
		TensorShape output_shape = x_tensor.shape();

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 1>(x_tensor);
		auto output = getDTensorTensorflow<T, 1>(*output_tensor);
		
		optox::ScaleOperator<T> op;
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}
};

#define REGISTER_GPU(dtype) \
	REGISTER_KERNEL_BUILDER( \
		Name("ScaleGrad") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<optox::tf<dtype>::type>("T"), \
		TFScaleOperator<dtype>) \

REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(float2);
REGISTER_GPU(double2);

#undef REGISTER_GPU