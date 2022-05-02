///@file th_maxpooling_operator.cpp
///@brief PyTorch wrappers for maxpooling operator
///@author Kaijie Mo <mokaijie5@gmail.com>
///@date 11.2021

#include <vector>

#include "th_utils.h"
#include "operators/maxpooling_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>

template<typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> forward1d(optox::MaxPooling1d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag)
{
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    int height_out_= op.calculateHeightOut1D();
    if (th_x_real.ndimension() == 2) {
    auto input_real = getDTensorTorch<T, 2>(th_x_real);
    auto input_imag = getDTensorTorch<T, 2>(th_x_imag); 

    // allocate the output tensor
    std::vector<int64_t> out_shape;
    out_shape.push_back(height_out_); // out height
    out_shape.push_back(in_shape[1]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_real.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 2>(th_out_real);
    auto output_imag = getDTensorTorch<T, 2>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 2>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag, th_out_idx);
    }
    else if (th_x_real.ndimension() == 3)
    {
    auto input_real = getDTensorTorch<T, 3>(th_x_real);
    auto input_imag = getDTensorTorch<T, 3>(th_x_imag); 

    // allocate the output tensor
    std::vector<int64_t> out_shape;
    out_shape.push_back(in_shape[0]); 
    out_shape.push_back(height_out_); // out height
    out_shape.push_back(in_shape[2]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_imag.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 3>(th_out_real);
    auto output_imag = getDTensorTorch<T, 3>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 3>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag,th_out_idx);
    }
    
}

template<typename T>
std::tuple<at::Tensor, at::Tensor> adjoint1d(optox::MaxPooling1d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag, at::Tensor th_output_real, at::Tensor th_output_imag, at::Tensor th_top_diff_real, at::Tensor th_top_diff_imag, at::Tensor indices)
{
int with_indices=op.getIndices();
if (with_indices==0){
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    int height_out_=op.calculateHeightOut1D();
    if (th_x_real.ndimension() == 2) {
    auto input_real = getDTensorTorch<T, 2>(th_x_real);
    auto input_imag = getDTensorTorch<T, 2>(th_x_imag);
    auto output_real = getDTensorTorch<T, 2>(th_output_real);
    auto output_imag = getDTensorTorch<T, 2>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 2>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 2>(th_top_diff_imag);

    
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 2>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 2>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (th_x_real.ndimension() == 3) {
    auto input_real = getDTensorTorch<T, 3>(th_x_real);
    auto input_imag = getDTensorTorch<T, 3>(th_x_imag);
    auto output_real = getDTensorTorch<T, 3>(th_output_real);
    auto output_imag = getDTensorTorch<T, 3>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 3>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 3>(th_top_diff_imag);
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 3>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 3>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
   else{// indices!=NULL
        // parse the tensors
    auto in_shape = indices.sizes().vec();
    if (indices.ndimension() == 2) {
    auto input_indices = getDTensorTorch<T, 2>(indices);
    auto top_diff_real = getDTensorTorch<T, 2>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 2>(th_top_diff_imag);
    in_shape[0]=op.getHeightIn1D();
    
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 2>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 2>(th_out_imag);
    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (indices.ndimension() == 3) {
    auto input_indices = getDTensorTorch<T, 3>(indices);
    auto top_diff_real = getDTensorTorch<T, 3>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 3>(th_top_diff_imag);
    in_shape[1]=op.getHeightIn1D();
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 3>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 3>(th_out_imag);
    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
    
    }
    
    



template<typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor> forward2d(optox::MaxPooling2d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag)
{
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    int height_out_= op.calculateHeightOut2D();
    int width_out_=op.calculateWidthOut2D();
    if (th_x_real.ndimension() == 3) {
    auto input_real = getDTensorTorch<T, 3>(th_x_real);
    auto input_imag = getDTensorTorch<T, 3>(th_x_imag); 

    // allocate the output tensor
    std::vector<int64_t> out_shape;

    out_shape.push_back(height_out_); // out height
    out_shape.push_back(width_out_);  // out weight
    out_shape.push_back(in_shape[2]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_real.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 3>(th_out_real);
    auto output_imag = getDTensorTorch<T, 3>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 3>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag, th_out_idx);
    }
    else if (th_x_real.ndimension() == 4)
    {
    auto input_real = getDTensorTorch<T, 4>(th_x_real);
    auto input_imag = getDTensorTorch<T, 4>(th_x_imag); 

    // allocate the output tensor
    std::vector<int64_t> out_shape;
    out_shape.push_back(in_shape[0]); 
    out_shape.push_back(height_out_); // out height
    out_shape.push_back(width_out_);  // out weight
    out_shape.push_back(in_shape[3]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_imag.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 4>(th_out_real);
    auto output_imag = getDTensorTorch<T, 4>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 4>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag,th_out_idx);
    }
    
}

template<typename T>
std::tuple<at::Tensor, at::Tensor> adjoint2d(optox::MaxPooling2d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag, at::Tensor th_output_real, at::Tensor th_output_imag, at::Tensor th_top_diff_real, at::Tensor th_top_diff_imag, at::Tensor indices)
{
int with_indices=op.getIndices();
if (with_indices==0){
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    int height_out_=op.calculateHeightOut2D();
    int width_out_=op.calculateWidthOut2D();
    if (th_x_real.ndimension() == 3) {
    auto input_real = getDTensorTorch<T, 3>(th_x_real);
    auto input_imag = getDTensorTorch<T, 3>(th_x_imag);
    auto output_real = getDTensorTorch<T, 3>(th_output_real);
    auto output_imag = getDTensorTorch<T, 3>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 3>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 3>(th_top_diff_imag);

    
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 3>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 3>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (th_x_real.ndimension() == 4) {
    auto input_real = getDTensorTorch<T, 4>(th_x_real);
    auto input_imag = getDTensorTorch<T, 4>(th_x_imag);
    auto output_real = getDTensorTorch<T, 4>(th_output_real);
    auto output_imag = getDTensorTorch<T, 4>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 4>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 4>(th_top_diff_imag);
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 4>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 4>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
   else{// indices!=NULL
        // parse the tensors
    auto in_shape = indices.sizes().vec();
    if (indices.ndimension() == 3) {
    auto input_indices = getDTensorTorch<T, 3>(indices);
    auto top_diff_real = getDTensorTorch<T, 3>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 3>(th_top_diff_imag);
    in_shape[0]=op.getHeightIn2D();
    in_shape[1]=op.getWidthIn2D();
    
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 3>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 3>(th_out_imag);
    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (indices.ndimension() == 4) {
    auto input_indices = getDTensorTorch<T, 4>(indices);
    auto top_diff_real = getDTensorTorch<T, 4>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 4>(th_top_diff_imag);
    in_shape[1]=op.getHeightIn2D();
    in_shape[2]=op.getWidthIn2D();
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 4>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 4>(th_out_imag);
    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
    
    }
    
    


template<typename T>
std::tuple<at::Tensor, at::Tensor,at::Tensor> forward3d(optox::MaxPooling3d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag)
{
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    int height_out_=op.calculateHeightOut3D();
    int width_out_=op.calculateWidthOut3D();
    int depth_out_=op.calculateDepthOut3D();
    if (th_x_real.ndimension() == 4) {
    auto input_real = getDTensorTorch<T, 4>(th_x_real);
    auto input_imag = getDTensorTorch<T, 4>(th_x_imag); 

    // allocate the output tensor
    std::vector<int64_t> out_shape;
    out_shape.push_back(height_out_); // out height
    out_shape.push_back(width_out_);  // out weight
    out_shape.push_back(depth_out_);  // out depth
    out_shape.push_back(in_shape[3]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_imag.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 4>(th_out_real);
    auto output_imag = getDTensorTorch<T, 4>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 4>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag, th_out_idx);
    }
    else if (th_x_real.ndimension() == 5)
    {
    auto input_real = getDTensorTorch<T, 5>(th_x_real);
    auto input_imag = getDTensorTorch<T, 5>(th_x_imag); 
    // allocate the output tensor
    std::vector<int64_t> out_shape;
    out_shape.push_back(in_shape[0]); 
    out_shape.push_back(height_out_); // out height
    out_shape.push_back(width_out_);  // out weight
    out_shape.push_back(depth_out_);  // out depth
    out_shape.push_back(in_shape[4]); // out channels
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_imag.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());
    auto output_real = getDTensorTorch<T, 5>(th_out_real);
    auto output_imag = getDTensorTorch<T, 5>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 5>(th_out_idx);
    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
    return std::make_tuple(th_out_real, th_out_imag,th_out_idx);
    }
    
}

template<typename T>
std::tuple<at::Tensor, at::Tensor> adjoint3d(optox::MaxPooling3d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag, at::Tensor th_output_real, at::Tensor th_output_imag, at::Tensor th_top_diff_real, at::Tensor th_top_diff_imag, at::Tensor indices)

{  int with_indices=op.getIndices(); 
if (with_indices==0){
    // parse the tensors
    auto in_shape = th_x_real.sizes().vec();
    if (th_x_real.ndimension() == 4) {
    auto input_real = getDTensorTorch<T, 4>(th_x_real);
    auto input_imag = getDTensorTorch<T, 4>(th_x_imag);
    auto output_real = getDTensorTorch<T, 4>(th_output_real);
    auto output_imag = getDTensorTorch<T, 4>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 4>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 4>(th_top_diff_imag);

    
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 4>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 4>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (th_x_real.ndimension() == 5) {
    auto input_real = getDTensorTorch<T, 5>(th_x_real);
    auto input_imag = getDTensorTorch<T, 5>(th_x_imag);
    auto output_real = getDTensorTorch<T, 5>(th_output_real);
    auto output_imag = getDTensorTorch<T, 5>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 5>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 5>(th_top_diff_imag);

    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 5>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 5>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
    else{// indices!=NULL
        // parse the tensors
    auto in_shape = indices.sizes().vec();
    
    if (indices.ndimension() == 4) {
    auto input_indices = getDTensorTorch<T, 4>(indices);
    auto top_diff_real = getDTensorTorch<T, 4>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 4>(th_top_diff_imag);
    in_shape[0]=op.getHeightIn3D();
    in_shape[1]=op.getWidthIn3D();
    in_shape[2]=op.getDepthIn3D();
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 4>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 4>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    else if (indices.ndimension() == 5) {
    auto input_indices = getDTensorTorch<T, 5>(indices);
    auto top_diff_real = getDTensorTorch<T, 5>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 5>(th_top_diff_imag);
    in_shape[1]=op.getHeightIn3D();
    in_shape[2]=op.getWidthIn3D();
    in_shape[3]=op.getDepthIn3D();

    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 5>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 5>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
    
    
    
    }
    
    

template<typename T>
std::tuple<at::Tensor, at::Tensor,at::Tensor, int, int, int, int> forward4d(optox::MaxPooling4d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag)
{
    // parse the tensors

    auto in_shape = th_x_real.sizes().vec();
    int time_out_=op.calculateTimeOut4D();
    int height_out_=op.calculateHeightOut4D();
    int width_out_=op.calculateWidthOut4D();
    int depth_out_=op.calculateDepthOut4D();
    int channel_out_=op.getChannels4D(); 
    int batch_out_=op.getBatch4D();  
    
    auto input_real = getDTensorTorch<T, 1>(th_x_real);
    auto input_imag = getDTensorTorch<T, 1>(th_x_imag); 
    // allocate the output tensor
    std::vector<int64_t> out_shape;
    if (batch_out_ !=0)
    	{out_shape.push_back(batch_out_ * time_out_ * height_out_* width_out_* depth_out_*channel_out_); }
    else {out_shape.push_back(time_out_ * height_out_* width_out_* depth_out_*channel_out_);}
    auto th_out_real = at::empty(out_shape, th_x_real.options());
    auto th_out_imag = at::empty(out_shape, th_x_imag.options());
    auto th_out_idx = at::empty(out_shape, th_x_real.options());

    auto output_real = getDTensorTorch<T, 1>(th_out_real);
    auto output_imag = getDTensorTorch<T, 1>(th_out_imag);
    auto output_idx = getDTensorTorch<T, 1>(th_out_idx);

    op.forward({output_real.get(), output_imag.get(), output_idx.get()}, {input_real.get(), input_imag.get()});
   
    return std::make_tuple(th_out_real, th_out_imag, th_out_idx, time_out_, height_out_, width_out_, depth_out_);

}

template<typename T>
std::tuple<at::Tensor, at::Tensor> adjoint4d(optox::MaxPooling4d_Operator<T> &op, at::Tensor th_x_real, at::Tensor th_x_imag, at::Tensor th_output_real, at::Tensor th_output_imag, at::Tensor th_top_diff_real, at::Tensor th_top_diff_imag, at::Tensor indices)

{  int with_indices=op.getIndices(); 
if (with_indices==0){
    // parse the tensors, with batch & without batch
    auto in_shape = th_x_real.sizes().vec();
    auto input_real = getDTensorTorch<T, 1>(th_x_real);
    auto input_imag = getDTensorTorch<T, 1>(th_x_imag);
    auto output_real = getDTensorTorch<T, 1>(th_output_real);
    auto output_imag = getDTensorTorch<T, 1>(th_output_imag);
    auto top_diff_real = getDTensorTorch<T, 1>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 1>(th_top_diff_imag);
   
    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, th_x_real.options());
    auto th_out_imag = at::empty(in_shape, th_x_imag.options());
    auto bottom_diff_real = getDTensorTorch<T, 1>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 1>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()}, {input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);

    }
    else{// indices!=NULL
        // parse the tensors
    auto in_shape = indices.sizes().vec();
    
    auto input_indices = getDTensorTorch<T, 1>(indices);
    auto top_diff_real = getDTensorTorch<T, 1>(th_top_diff_real);
    auto top_diff_imag = getDTensorTorch<T, 1>(th_top_diff_imag);
    int b_in = op.getBatch4D();
    int t_in = op.getTimeIn4D();
    int h_in = op.getHeightIn4D();
    int w_in = op.getWidthIn4D();
    int d_in = op.getDepthIn4D();
    int ch_in =op.getChannels4D();
    
    if (b_in!=0){
    in_shape[0]=b_in * t_in * h_in*  w_in * d_in *ch_in;
    }
    else{
    in_shape[0]=t_in * h_in* w_in * d_in *ch_in;
    }

    // allocate the output tensor
    auto th_out_real = at::empty(in_shape, indices.options());
    auto th_out_imag = at::empty(in_shape, indices.options());
    auto bottom_diff_real = getDTensorTorch<T, 1>(th_out_real);
    auto bottom_diff_imag = getDTensorTorch<T, 1>(th_out_imag);

    op.adjoint({bottom_diff_real.get(),bottom_diff_imag.get()},{input_indices.get(), top_diff_real.get(), top_diff_imag.get()});

    return std::make_tuple(th_out_real, th_out_imag);
    }
    }
    
    
    
    
    
    
      

template<typename T>
void declare_op(py::module &m, const std::string &typestr)

{       
    std::string pyclass_name_1d = std::string("MaxPooling1d_") + typestr;
    py::class_<optox::MaxPooling1d_Operator<T>>(m, pyclass_name_1d.c_str(), py::buffer_protocol(), py::dynamic_attr())			
    .def(py::init<int&,   int&,   int&,    int&, float&, float&,     int&,   int&,    int&, int&,int&,  std::string&>())
    .def("forward", forward1d<T>)
    .def("adjoint", adjoint1d<T>);
    
    
    std::string pyclass_name_2d = std::string("MaxPooling2d_") + typestr;
    py::class_<optox::MaxPooling2d_Operator<T>>(m, pyclass_name_2d.c_str(), py::buffer_protocol(), py::dynamic_attr())			
    .def(py::init<int&,int&,   int&,int&,   int&,int&,    int&, float&, float&,     int&,int&,   int&, int&,    int&, int&,int&,  std::string&>())
    .def("forward", forward2d<T>)
    .def("adjoint", adjoint2d<T>);

    std::string pyclass_name_3d = std::string("MaxPooling3d_") + typestr;
    py::class_<optox::MaxPooling3d_Operator<T>>(m, pyclass_name_3d.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int&,int&,int&,   int&, int&, int&,    int&, int&, int&,    int&, float&, float&,    int&, int&,int&,   int&, int&, int&,     int&,int&,int&,  std::string&>())
    .def("forward", forward3d<T>)
    .def("adjoint", adjoint3d<T>);
    
    std::string pyclass_name_4d = std::string("MaxPooling4d_") + typestr;
    py::class_<optox::MaxPooling4d_Operator<T>>(m, pyclass_name_4d.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int&,int&,int&,int&, 
                 int&, int&, int&,int&,
                 int&, int&, int&,int&, 
                  int&, float&, float&,    
                  int&, int&,int&,int&,    
                  int&, int&, int&,int&,   
                    int&,int&, int&, std::string&>())
    .def("forward", forward4d<T>)
    .def("adjoint", adjoint4d<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
