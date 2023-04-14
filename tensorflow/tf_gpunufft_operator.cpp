///@file tf_gpunufft_operator.cpp
///@brief tensorflow wrappers for the gpunufft operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 16.03.2020

#include "tf_utils.h"
#include "operators/gpunufft_operator.h"
#include "typetraits.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>

using namespace tensorflow;
using namespace std;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status GpuNufftShapeFn(InferenceContext* c) {
    // We can infer the output shape for the GpuNufft (forward) operator
    // from cmap and trajectory. Make safety check that input dimensionality
    // are of rank 3.
    ShapeHandle cmap = c->input(1);
    TF_RETURN_IF_ERROR( c->WithRank(cmap, 3, &cmap));
    ShapeHandle trajectory = c->input(2);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory, 3, &trajectory));
    // create Dimension Handle to store output dimensions
    std::vector<DimensionHandle> dims(3);
    // Take number of frames of trajectory
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory, 0), 0     , &dims[0]) );
    // ...and number of kspace points...
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory, 2), 0     , &dims[2]) );
    // ...and number of coils from cmap.
    TF_RETURN_IF_ERROR( c->Add( c->Dim( cmap, 0), 0     , &dims[1]) );

    // Finally make output shape
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
}

Status GpuNufftAdjointShapeFn(InferenceContext* c) {
    // We can infer the output shape for the GpuNufft (adjoint) operator
    // from the attribute img_dim and the trajectory. Make safety check that input dimensionality is of rank 3.
    int img_dim;
    c->GetAttr("img_dim", &img_dim);
    ShapeHandle trajectory = c->input(2);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory, 3, &trajectory));

    // Create image shape of img_dim
    ShapeHandle image;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(img_dim), c->Vector(img_dim), &image));
    // ... and concatenate it with number of frames at dim=0
    ShapeHandle output;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->Dim(trajectory, 0)), image, &output));

    // Finally make output shape
    c->set_output(0, output);
    return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("GpuNufftOperatorForward")
		.Input("image: complex64")
		.Input("sensitivities: complex64")
		.Input("trajectory: T")
		.Input("dcf: T")
		.Output("rawdata: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftShapeFn);

REGISTER_OP("GpuNufftOperatorAdjoint")
		.Input("rawdata: complex64")
		.Input("sensitivities: complex64")
		.Input("trajectory: T")
		.Input("dcf: T")
		.Output("image: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftAdjointShapeFn);

template <typename T>
class TFGpuNufftForward : public OpKernel
{
  public:
    explicit TFGpuNufftForward(OpKernelConstruction *context)
        : OpKernel(context)
	{
        // Init attributes
        // T osf = 0.0;
        // int img_dim = 0;
        // int kernel_width = 0;
        // int sector_width = 0;

		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

        //op_ = new optox::GPUNufftOperator<T>(img_dim, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftForward()
    {
        // if (op_)
        //     delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& img_tensor = context->input(0);
		const Tensor& csm_tensor = context->input(1);
		const Tensor& traj_tensor = context->input(2);
		const Tensor& dcf_tensor = context->input(3);

        // Prepare output shape
		auto rawdata_shape = traj_tensor.shape();
		auto dims = traj_tensor.dims();
		auto csm_dims = csm_tensor.dims();
		rawdata_shape.set_dim(dims-2, csm_tensor.dim_size(csm_dims-3));

        // // Check dimensionality
        // OP_REQUIRES(context, input_tensor.dims() == 4,
        //             errors::Unimplemented("Expected a 4d Tensor, got ",
        //                                   input_tensor.dims(), "d."));

        // OP_REQUIRES(context, input_tensor.dim_size(3) == 3,
        //             errors::Unimplemented("Expected the channel dimension to be 3, got ",
        //                                   input_tensor.dim_size(3), "."));

		// Allocate the output
		Tensor* rawdata_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, rawdata_shape, &rawdata_tensor));
 
        // compute the output
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(*rawdata_tensor);
        auto img = getComplexDTensorTensorflow<complex_type, 3>(img_tensor);
        auto csm = getComplexDTensorTensorflow<complex_type, 3>(csm_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(context->eigen_device<GPUDevice>().stream());
        op.forward({rawdata.get()}, {img.get(), csm.get(), trajectory.get(), dcf.get()});
    }

  private:
    //optox::GPUNufftOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftOperatorForward") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftForward<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU

template <typename T>
class TFGpuNufftAdjoint : public OpKernel
{
  public:
    explicit TFGpuNufftAdjoint(OpKernelConstruction *context)
        : OpKernel(context)
	{
        // Init attributes
        // T osf = 0.0;
        // int kernel_width = 0;
        // int sector_width = 0;

		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

        //op_ = new optox::GPUNufftOperator<T>(img_dim_, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftAdjoint()
    {
        // if (op_)
        //     delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& rawdata_tensor = context->input(0);
		const Tensor& csm_tensor = context->input(1);
		const Tensor& traj_tensor = context->input(2);
		const Tensor& dcf_tensor = context->input(3);
 
		// TODO size checks, dim checks!
        // // Check dimensionality
        // OP_REQUIRES(context, input_tensor.dims() == 4,
        //             errors::Unimplemented("Expected a 4d Tensor, got ",
        //                                   input_tensor.dims(), "d."));

        // OP_REQUIRES(context, input_tensor.dim_size(3) == 1,
        //             errors::Unimplemented("Expected the channel dimension to be 1, got ",
        //                                   input_tensor.dim_size(3), "."));

		// Prepare output shape
		auto img_shape = rawdata_tensor.shape();
		auto dims = rawdata_tensor.dims();
		img_shape.set_dim(dims-2, img_dim_);
		img_shape.set_dim(dims-1, img_dim_);

		// Allocate the output
		Tensor* img_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, img_shape, &img_tensor));
 
        // compute the output
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto img = getComplexDTensorTensorflow<complex_type, 3>(*img_tensor);
        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(rawdata_tensor);
        auto csm = getComplexDTensorTensorflow<complex_type, 3>(csm_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(context->eigen_device<GPUDevice>().stream());
        op.adjoint({img.get()}, {rawdata.get(), csm.get(), trajectory.get(), dcf.get()});
    }

  private:
    //optox::GPUNufftOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftOperatorAdjoint") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftAdjoint<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU

// Singlecoil GpuNufft
Status GpuNufftSinglecoilShapeFn(InferenceContext* c) {
    // We can infer the output shape for the GpuNufft (forward) operator
    // from the dcf. Make safety check that input dimensionality
    // are of rank 3.
    ShapeHandle dcf = c->input(2);
    TF_RETURN_IF_ERROR( c->WithRank(dcf, 3, &dcf));

    // Finally make output shape
    c->set_output(0, dcf);
    return Status::OK();
}

Status GpuNufftSinglecoilAdjointShapeFn(InferenceContext* c) {
    // We can infer the output shape for the GpuNufft (adjoint) operator
    // from the attribute img_dim and the trajectory. Make safety check that input dimensionality is of rank 3.
    int img_dim;
    c->GetAttr("img_dim", &img_dim);
    ShapeHandle trajectory = c->input(1);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory, 3, &trajectory));

    // Create image shape of img_dim
    ShapeHandle image;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(img_dim), c->Vector(img_dim), &image));
    // ... and concatenate it with number of frames at dim=0
    ShapeHandle output;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->Dim(trajectory, 0)), image, &output));

    // Finally make output shape
    c->set_output(0, output);
    return Status::OK();
}
/**
 * register the operation with necessary options
 */
REGISTER_OP("GpuNufftSingleCoilOperatorForward")
		.Input("image: complex64")
		.Input("trajectory: T")
		.Input("dcf: T")
		.Output("rawdata: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftSinglecoilShapeFn);

REGISTER_OP("GpuNufftSingleCoilOperatorAdjoint")
		.Input("rawdata: complex64")
		.Input("trajectory: T")
		.Input("dcf: T")
		.Output("image: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftSinglecoilAdjointShapeFn);

template <typename T>
class TFGpuNufftSingleCoilForward : public OpKernel
{
  public:
    explicit TFGpuNufftSingleCoilForward(OpKernelConstruction *context)
        : OpKernel(context)
	{
		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

        //op_ = new optox::GPUNufftSingleCoilOperator<T>(img_dim, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftSingleCoilForward()
    {
        // if (op_)
        //     delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& img_tensor = context->input(0);
		const Tensor& traj_tensor = context->input(1);
		const Tensor& dcf_tensor = context->input(2);

        // Check dimensionality
        OP_REQUIRES(context, img_tensor.dims() == 3,
                    errors::Unimplemented("Img: Expected a 3d Tensor, got ",
                                          img_tensor.dims(), "d."));
        OP_REQUIRES(context, traj_tensor.dims() == 3,
                    errors::Unimplemented("Traj: Expected a 3d Tensor, got ",
                                          traj_tensor.dims(), "d."));
        OP_REQUIRES(context, dcf_tensor.dims() == 3,
                    errors::Unimplemented("Dcf: Expected a 3d Tensor, got ",
                                          dcf_tensor.dims(), "d."));

        // Prepare output shape
		auto rawdata_shape = dcf_tensor.shape();

		// Allocate the output
		Tensor* rawdata_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, rawdata_shape, &rawdata_tensor));
 
        // compute the output
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(*rawdata_tensor);
        auto img = getComplexDTensorTensorflow<complex_type, 3>(img_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftSingleCoilOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(context->eigen_device<GPUDevice>().stream());
        op.forward({rawdata.get()}, {img.get(), trajectory.get(), dcf.get()});
    }

  private:
    // optox::GPUNufftSingleCoilOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftSingleCoilOperatorForward") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftSingleCoilForward<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU

template <typename T>
class TFGpuNufftSingleCoilAdjoint : public OpKernel
{
  public:
    explicit TFGpuNufftSingleCoilAdjoint(OpKernelConstruction *context)
        : OpKernel(context)
	{
		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

       // op_ = new optox::GPUNufftSingleCoilOperator<T>(img_dim_, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftSingleCoilAdjoint()
    {
        // if (op_)
        //     delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& rawdata_tensor = context->input(0);
		const Tensor& traj_tensor = context->input(1);
		const Tensor& dcf_tensor = context->input(2);
 
        // Check dimensionality
        OP_REQUIRES(context, rawdata_tensor.dims() == 3,
                    errors::Unimplemented("Expected a 3d Tensor, got ",
                                          rawdata_tensor.dims(), "d."));

        OP_REQUIRES(context, traj_tensor.dims() == 3,
                    errors::Unimplemented("Expected a 3d Tensor, got ",
                                          traj_tensor.dims(), "d."));

        OP_REQUIRES(context, dcf_tensor.dims() == 3,
                    errors::Unimplemented("Expected a 3d Tensor, got ",
                                          dcf_tensor.dims(), "d."));

		// Prepare output shape
	    TensorShape img_shape({rawdata_tensor.dim_size(0), img_dim_, img_dim_});
		// Allocate the output
		Tensor* img_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, img_shape, &img_tensor));
 
        // compute the output
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto img = getComplexDTensorTensorflow<complex_type, 3>(*img_tensor);
        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(rawdata_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftSingleCoilOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(context->eigen_device<GPUDevice>().stream());
        op.adjoint({img.get()}, {rawdata.get(), trajectory.get(), dcf.get()});
    }

  private:
    //optox::GPUNufftSingleCoilOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftSingleCoilOperatorAdjoint") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftSingleCoilAdjoint<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU

// **** MULTIRES GPUNUFFT OPERATOR EXPERIMENTA ****

Status GpuNufftMultiresShapeFn(InferenceContext* c) {
    // Input 0: img1
    // Input 1: traj1
    // Input 2: dcf1
    // Input 3: img2
    // Input 4: traj2
    // Input 5: dcf2
    // Input 6: cmap

    // We can infer the output shape for the GpuNufftMultires (forward) operator
    // from cmap and trajectory. Make safety check that input dimensionality
    // are of rank 3.
    ShapeHandle cmap = c->input(6);
    TF_RETURN_IF_ERROR( c->WithRank(cmap, 3, &cmap));

    // First input
    ShapeHandle trajectory1 = c->input(1);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory1, 3, &trajectory1));
    // create Dimension Handle to store output dimensions
    std::vector<DimensionHandle> dims1(3);
    // Take number of frames of trajectory
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory1, 0), 0     , &dims1[0]) );
    // ...and number of kspace points...
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory1, 2), 0     , &dims1[2]) );
    // ...and number of coils from cmap.
    TF_RETURN_IF_ERROR( c->Add( c->Dim( cmap, 0), 0     , &dims1[1]) );

    // Second input
    ShapeHandle trajectory2 = c->input(4);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory2, 3, &trajectory2));
    // create Dimension Handle to store output dimensions
    std::vector<DimensionHandle> dims2(3);
    // Take number of frames of trajectory
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory2, 0), 0     , &dims2[0]) );
    // ...and number of kspace points...
    TF_RETURN_IF_ERROR( c->Add( c->Dim( trajectory2, 2), 0     , &dims2[2]) );
    // ...and number of coils from cmap.
    TF_RETURN_IF_ERROR( c->Add( c->Dim( cmap, 0), 0     , &dims2[1]) );

    // Finally make output shape
    c->set_output(0, c->MakeShape(dims1));
    c->set_output(1, c->MakeShape(dims2));
    return Status::OK();
}

Status GpuNufftMultiresAdjointShapeFn(InferenceContext* c) {
    // We can infer the output shape for the GpuNufftMultires (adjoint) operator
    // from the attribute img_dim and the trajectory. Make safety check that input dimensionality is of rank 3.
    int img_dim;
    c->GetAttr("img_dim", &img_dim);
    ShapeHandle trajectory1 = c->input(1);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory1, 3, &trajectory1));
    ShapeHandle trajectory2 = c->input(4);
    TF_RETURN_IF_ERROR( c->WithRank(trajectory2, 3, &trajectory2));

    // Create image shape of img_dim
    ShapeHandle image;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(img_dim), c->Vector(img_dim), &image));
    // ... and concatenate it with number of frames at dim=0
    ShapeHandle output1;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->Dim(trajectory1, 0)), image, &output1));
    ShapeHandle output2;
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->Dim(trajectory2, 0)), image, &output2));

    // Finally make output shape
    c->set_output(0, output1);
    c->set_output(1, output2);
    return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("GpuNufftMultiresOperatorForward")
		.Input("image1: complex64")
		.Input("trajectory1: T")
		.Input("dcf1: T")
        .Input("image2: complex64")
        .Input("trajectory2: T")
        .Input("dcf2: T")
		.Input("sensitivities: complex64")
		.Output("rawdata1: complex64")
        .Output("rawdata2: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftMultiresShapeFn);

REGISTER_OP("GpuNufftMultiresOperatorAdjoint")
		.Input("rawdata1: complex64")
		.Input("trajectory1: T")
		.Input("dcf1: T")
        .Input("rawdata2: complex64")
        .Input("trajectory2: T")
        .Input("dcf2: T")
		.Input("sensitivities: complex64")
		.Output("image1: complex64")
		.Output("image2: complex64")
        .Attr("T: {float32}")
		.Attr("osf: float")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(GpuNufftMultiresAdjointShapeFn);

template <typename T>
class TFGpuNufftMultiresForward : public OpKernel
{
  public:
    explicit TFGpuNufftMultiresForward(OpKernelConstruction *context)
        : OpKernel(context)
	{
        // Init attributes
        // T osf = 0.0;
        // int img_dim = 0;
        // int kernel_width = 0;
        // int sector_width = 0;

		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

        // op_ = new optox::GPUNufftOperator<T>(img_dim, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftMultiresForward()
    {
        // if (op_)
        //     delete op_;
    }

    void apply_operator(const Tensor& img_tensor, const Tensor& traj_tensor,
        const Tensor& dcf_tensor, const Tensor& csm_tensor, Tensor* rawdata_tensor, const cudaStream_t& stream)
    {
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(*rawdata_tensor);
        auto img = getComplexDTensorTensorflow<complex_type, 3>(img_tensor);
        auto csm = getComplexDTensorTensorflow<complex_type, 3>(csm_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(stream);
        op.forward({rawdata.get()}, {img.get(), csm.get(), trajectory.get(), dcf.get()});
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& img1_tensor = context->input(0);
		const Tensor& traj1_tensor = context->input(1);
		const Tensor& dcf1_tensor = context->input(2);
		const Tensor& img2_tensor = context->input(3);
		const Tensor& traj2_tensor = context->input(4);
		const Tensor& dcf2_tensor = context->input(5);
		const Tensor& csm_tensor = context->input(6);

        // Prepare output shape
		auto rawdata1_shape = traj1_tensor.shape();
		auto dims1 = traj1_tensor.dims();
		auto csm_dims = csm_tensor.dims();
		rawdata1_shape.set_dim(dims1-2, csm_tensor.dim_size(csm_dims-3));

		auto rawdata2_shape = traj2_tensor.shape();
		auto dims2 = traj2_tensor.dims();
		rawdata2_shape.set_dim(dims2-2, csm_tensor.dim_size(csm_dims-3));

        // // Check dimensionality
        // OP_REQUIRES(context, input_tensor.dims() == 4,
        //             errors::Unimplemented("Expected a 4d Tensor, got ",
        //                                   input_tensor.dims(), "d."));

        // OP_REQUIRES(context, input_tensor.dim_size(3) == 3,
        //             errors::Unimplemented("Expected the channel dimension to be 3, got ",
        //                                   input_tensor.dim_size(3), "."));

		// Allocate the output
		Tensor* rawdata1_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, rawdata1_shape, &rawdata1_tensor));
 		Tensor* rawdata2_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(1, rawdata2_shape, &rawdata2_tensor));

        // compute the output
        auto stream = context->eigen_device<GPUDevice>().stream();
        apply_operator(img1_tensor, traj1_tensor, dcf1_tensor, csm_tensor, rawdata1_tensor, stream);
        apply_operator(img2_tensor, traj2_tensor, dcf2_tensor, csm_tensor, rawdata2_tensor, stream);
    }

  private:
    //optox::GPUNufftOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftMultiresOperatorForward") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftMultiresForward<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU

template <typename T>
class TFGpuNufftMultiresAdjoint : public OpKernel
{
  public:
    explicit TFGpuNufftMultiresAdjoint(OpKernelConstruction *context)
        : OpKernel(context)
	{
		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));

        //op_ = new optox::GPUNufftOperator<T>(img_dim_, osf, kernel_width, sector_width);
	}

    virtual ~TFGpuNufftMultiresAdjoint()
    {
        //if (op_)
        //    delete op_;
    }

    void apply_operator(const Tensor& rawdata_tensor, const Tensor& traj_tensor,
        const Tensor& dcf_tensor, const Tensor& csm_tensor, Tensor* img_tensor, const cudaStream_t& stream)
    {
        // compute the output
        typedef typename optox::type_trait<T>::complex_type complex_type;
        typedef typename optox::type_trait<T>::real_type real_type;

        auto img = getComplexDTensorTensorflow<complex_type, 3>(*img_tensor);
        auto rawdata = getComplexDTensorTensorflow<complex_type, 3>(rawdata_tensor);
        auto csm = getComplexDTensorTensorflow<complex_type, 3>(csm_tensor);
        auto trajectory = getDTensorTensorflow<real_type, 3>(traj_tensor);
        auto dcf = getDTensorTensorflow<real_type, 3>(dcf_tensor);

        optox::GPUNufftOperator<T> op(img_dim_, osf_, kernel_width_, sector_width_);
        op.setStream(stream);
        op.adjoint({img.get()}, {rawdata.get(), csm.get(), trajectory.get(), dcf.get()});
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensors
		const Tensor& rawdata1_tensor = context->input(0);
		const Tensor& traj1_tensor = context->input(1);
		const Tensor& dcf1_tensor = context->input(2);
		const Tensor& rawdata2_tensor = context->input(3);
		const Tensor& traj2_tensor = context->input(4);
		const Tensor& dcf2_tensor = context->input(5);
		const Tensor& csm_tensor = context->input(6);

		// TODO size checks, dim checks!
        // // Check dimensionality
        // OP_REQUIRES(context, input_tensor.dims() == 4,
        //             errors::Unimplemented("Expected a 4d Tensor, got ",
        //                                   input_tensor.dims(), "d."));

        // OP_REQUIRES(context, input_tensor.dim_size(3) == 1,
        //             errors::Unimplemented("Expected the channel dimension to be 1, got ",
        //                                   input_tensor.dim_size(3), "."));

		// Prepare output shape
		auto img1_shape = rawdata1_tensor.shape();
		auto dims1 = rawdata1_tensor.dims();
		img1_shape.set_dim(dims1-2, img_dim_);
		img1_shape.set_dim(dims1-1, img_dim_);

		// Prepare output shape
		auto img2_shape = rawdata2_tensor.shape();
		auto dims2 = rawdata2_tensor.dims();
		img2_shape.set_dim(dims2-2, img_dim_);
		img2_shape.set_dim(dims2-1, img_dim_);

		// Allocate the output
		Tensor* img1_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, img1_shape, &img1_tensor));
 		Tensor* img2_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(1, img2_shape, &img2_tensor));

        // compute the output
        auto stream = context->eigen_device<GPUDevice>().stream();
        apply_operator(rawdata1_tensor, traj1_tensor, dcf1_tensor, csm_tensor, img1_tensor, stream);
        apply_operator(rawdata2_tensor, traj2_tensor, dcf2_tensor, csm_tensor, img2_tensor, stream);
    }

  private:
    //optox::GPUNufftOperator<T> *op_ = nullptr;
    int img_dim_;
    T osf_;
    int kernel_width_;
    int sector_width_;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("GpuNufftMultiresOperatorAdjoint") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        TFGpuNufftMultiresAdjoint<T>)

REGISTER_GPU(float);

#undef REGISTER_GPU