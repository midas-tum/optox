#include "utils.h"
#include "tensor/d_tensor.h"
#include "maxpooling_operator.h"
#include "reduce.cuh"


template < typename T>
	__global__
void maxpooling1d_complex(	const typename optox::DTensor<T, 2>::ConstRef input_real, const typename optox::DTensor<T, 2>::ConstRef input_imag,
			typename optox::DTensor<T, 2>::Ref output_real,
			typename optox::DTensor<T, 2>::Ref output_imag,
			typename optox::DTensor<T, 2>::Ref output_idx,
			int height_in,
			int height_out,
			int kernel_h,
			int stride_h,
			int channels, float alpha, float beta,
			int pad_h)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int ch = threadIdx.y + blockIdx.y *blockDim.y;

	if (i < height_out && ch < channels)
	{
		T maxval_real = 0;
		T maxval_imag = 0;

		int maxidx = -1;
		int hstart = i * stride_h -pad_h;
		int hend = min(hstart + kernel_h, height_in);
		for (int h = hstart; h < hend; ++h)
		{

			T real = input_real(h, ch);
			T imag = input_imag(h, ch);

			T amp = alpha *real *real + beta *imag * imag;
			T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
			int idx = h;
			if (amp > max_value)
			{
				maxidx = idx;
				maxval_real = real;
				maxval_imag = imag;
			}

			output_real(i, ch) = maxval_real;
			output_imag(i, ch) = maxval_imag;
			output_idx(i, ch) = maxidx;
		}
	}
}
template < typename T>
	__global__
void maxpooling1d_complex_with_batch(	const typename optox::DTensor<T, 3>::ConstRef input_real, const typename optox::DTensor<T, 3>::ConstRef input_imag,
			typename optox::DTensor<T, 3>::Ref output_real,
			typename optox::DTensor<T, 3>::Ref output_imag,
			typename optox::DTensor<T, 3>::Ref output_idx,
			int height_in,
			int height_out,
			int kernel_h,
			int stride_h,
			int channels, float alpha, float beta,
			int pad_h, int batch = 1)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < batch && j < height_out && ch < channels)
	{

		T maxval_real = 0;
		T maxval_imag = 0;

		int maxidx = -1;
		int hstart = j * stride_h -pad_h;
		int hend = min(hstart + kernel_h, height_in);

		for (int h = hstart; h < hend; ++h)
		{

			T real = input_real(i, h, ch);
			T imag = input_imag(i, h, ch);

			T amp = alpha *real *real + beta *imag * imag;
			T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
			int idx = h;
			if (amp > max_value)
			{
				maxidx = idx;
				maxval_real = real;
				maxval_imag = imag;
			}

			output_real(i, j, ch) = maxval_real;
			output_imag(i, j, ch) = maxval_imag;
			output_idx(i, j, ch) = maxidx;
		
		}
	}
}

template < typename T>
	void optox::MaxPooling1d_Operator<T>::computeForward(		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs
)
	{
		if (!batch_)
		{
			auto in_real = this->template getInput<T, 2> (0, inputs);
			auto in_imag = this->template getInput<T, 2> (1, inputs);
			auto out_real = this->template getOutput<T, 2> (0, outputs);
			auto out_imag = this->template getOutput<T, 2> (1, outputs);
			auto out_idx = this->template getOutput<T, 2> (2, outputs);
			dim3 dim_block = dim3(16 * 16, 4);
			dim3 dim_grid = dim3(divUp(height_out_, dim_block.x), divUp(channels_, dim_block.y));

			maxpooling1d_complex<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, height_out_,
				kernel_h_,
				stride_h_,
				channels_, alpha_, beta_, pad_h_);

			OPTOX_CUDA_CHECK;
		}
		else
		{
			auto in_real = this->template getInput<T, 3> (0, inputs);
			auto in_imag = this->template getInput<T, 3> (1, inputs);
			auto out_real = this->template getOutput<T, 3> (0, outputs);
			auto out_imag = this->template getOutput<T, 3> (1, outputs);
			auto out_idx = this->template getOutput<T, 3> (2, outputs);
			dim3 dim_block = dim3(16, 8, 4);
			dim3 dim_grid = dim3(divUp(batch_, dim_block.x),
				divUp(height_out_, dim_block.y),
				divUp(channels_, dim_block.z)
		);

			maxpooling1d_complex_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, height_out_,
				kernel_h_,
				stride_h_,
				channels_, alpha_, beta_, pad_h_, batch_);

			OPTOX_CUDA_CHECK;
		}
	}

template < typename T>
	__global__
void maxpooling1d_complex_grad_backward(	const typename optox::DTensor<T, 2>::ConstRef input_real, const typename optox::DTensor<T, 2>::ConstRef input_imag, const typename optox::DTensor<T, 2>::ConstRef output_real, const typename optox::DTensor<T, 2>::ConstRef output_imag, const typename optox::DTensor<T, 2>::ConstRef top_diff_real, const typename optox::DTensor<T, 2>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 2>::Ref bottom_diff_real,
							typename optox::DTensor<T, 2>::Ref bottom_diff_imag,
							int height_in,
							int height_out,
							int kernel_h,
							int stride_h,
							int channels, float alpha, float beta,
							int pad_h, int n = 0)
{

	for (int h = 0; h < height_in; h++)
		for (int ch = 0; ch < channels; ch++)
		{
			bottom_diff_real(h, ch) = 0;
			bottom_diff_imag(h, ch) = 0;
		}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int ch = threadIdx.y + blockIdx.y *blockDim.y;

	if (i < height_out && ch < channels)
	{
		int maxidx_h = -1;
		int hstart = i * stride_h - pad_h;
		int hend = min(hstart + kernel_h, height_in);
		bool should_stop = false;
		for (int h = hstart; h < hend && !should_stop; ++h)
		{
			if (output_real(i, ch) == input_real(h, ch) && output_imag(i, ch) == input_imag(h, ch))
			{
				maxidx_h = h;
				should_stop = true;
			}
		}
		if (maxidx_h != -1)
		{
			bottom_diff_real(maxidx_h, ch) = top_diff_real(i, ch);
			bottom_diff_imag(maxidx_h, ch) = top_diff_imag(i, ch);
		}
	}
}

template < typename T>
	__global__
void maxpooling1d_complex_grad_backward_with_batch(	const typename optox::DTensor<T, 3>::ConstRef input_real, const typename optox::DTensor<T, 3>::ConstRef input_imag, const typename optox::DTensor<T, 3>::ConstRef output_real, const typename optox::DTensor<T, 3>::ConstRef output_imag, const typename optox::DTensor<T, 3>::ConstRef top_diff_real, const typename optox::DTensor<T, 3>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 3>::Ref bottom_diff_real,
							typename optox::DTensor<T, 3>::Ref bottom_diff_imag,
							int height_in,
							int height_out,
							int kernel_h,
							int stride_h,
							int channels, float alpha, float beta,
							int pad_h, int batch = 1)
{
	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int ch = 0; ch < channels; ch++)
			{
				bottom_diff_real(b, h, ch) = 0;
				bottom_diff_imag(b, h, ch) = 0;
			}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;
	if (i < batch && j < height_out && ch < channels)
	{
		int maxidx_h = -1;
		int hstart = j * stride_h - pad_h;
		int hend = min(hstart + kernel_h, height_in);

		bool should_stop = false;

		for (int h = hstart; h < hend && !should_stop; ++h)
		{
			if (output_real(i, j, ch) == input_real(i, h, ch) && output_imag(i, j, ch) == input_imag(i, h, ch))
			{
				maxidx_h = h;
				should_stop = true;
			}
		}

		if (maxidx_h != -1)
		{
			bottom_diff_real(i, maxidx_h, ch) = top_diff_real(i, j, ch);
			bottom_diff_imag(i, maxidx_h, ch) = top_diff_imag(i, j, ch);
		}
	}
}

template < typename T>
	__global__
void maxpooling1d_complex_grad_backward_indices(	const typename optox::DTensor<T, 2>::ConstRef indices, const typename optox::DTensor<T, 2>::ConstRef top_diff_real, const typename optox::DTensor<T, 2>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 2>::Ref bottom_diff_real,
				typename optox::DTensor<T, 2>::Ref bottom_diff_imag,
				int height_in,
				int height_out,
				int kernel_h,
				int stride_h,
				int channels, float alpha, float beta,
				int pad_h)
{

	for (int h = 0; h < height_in; h++)

		for (int ch = 0; ch < channels; ch++)
		{
			bottom_diff_real(h, ch) = 0;
			bottom_diff_imag(h, ch) = 0;
		}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int ch = threadIdx.y + blockIdx.y *blockDim.y;

	if (i < height_out && ch < channels)
	{
		int idx = indices(i, ch);
		int n = idx;
		int h = n % height_in;
		bottom_diff_real(h, ch) = top_diff_real(i, ch);
		bottom_diff_imag(h, ch) = top_diff_imag(i, ch);
	}
}

template < typename T>
	__global__
void maxpooling1d_complex_grad_backward_with_batch_and_indices(	const typename optox::DTensor<T, 3>::ConstRef indices, const typename optox::DTensor<T, 3>::ConstRef top_diff_real, const typename optox::DTensor<T, 3>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 3>::Ref bottom_diff_real,
				typename optox::DTensor<T, 3>::Ref bottom_diff_imag,
				int height_in,
				int height_out,
				int kernel_h,
				int stride_h,
				int channels, float alpha, float beta,
				int pad_h, int batch = 1)
{

	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int ch = 0; ch < channels; ch++)
			{
				bottom_diff_real(b, h,  ch) = 0;
				bottom_diff_imag(b, h,  ch) = 0;
			}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;
	if (i < batch && j < height_out && ch < channels)
	{

		int idx = indices(i, j, ch);
		int n = idx;
		int h = n % height_in;
		bottom_diff_real(i, h, ch) = top_diff_real(i, j, ch);
		bottom_diff_imag(i, h, ch) = top_diff_imag(i, j, ch);
	}
}

template < typename T>
	void optox::MaxPooling1d_Operator<T>::computeAdjoint(		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs

)
	{
		if (!with_indices_)
		{
			if (!batch_)
			{
				auto in_real = this->template getInput<T, 2> (0, inputs);
				auto in_imag = this->template getInput<T, 2> (1, inputs);

				auto out_real = this->template getInput<T, 2> (2, inputs);
				auto out_imag = this->template getInput<T, 2> (3, inputs);

				auto top_diff_real = this->template getInput<T, 2> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 2> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 2> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 2> (1, outputs);

				dim3 dim_block = dim3(16 * 16, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(channels_, dim_block.y));

				maxpooling1d_complex_grad_backward<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, height_out_,
					kernel_h_,
					stride_h_,
					channels_, alpha_, beta_, pad_h_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
			 	//batch=true
				auto in_real = this->template getInput<T, 3> (0, inputs);
				auto in_imag = this->template getInput<T, 3> (1, inputs);

				auto out_real = this->template getInput<T, 3> (2, inputs);
				auto out_imag = this->template getInput<T, 3> (3, inputs);

				auto top_diff_real = this->template getInput<T, 3> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 3> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 3> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 3> (1, outputs);

				dim3 dim_block = dim3(16, 16, 4);
				dim3 dim_grid = dim3(divUp(batch_, dim_block.x),
					divUp(height_out_, dim_block.y),
					divUp(channels_, dim_block.z));

				maxpooling1d_complex_grad_backward_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_,
					height_out_,
					kernel_h_,
					stride_h_,
					channels_, alpha_, beta_, pad_h_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
		else
		{
			//indices
			if (!batch_)
			{
				auto indices = this->template getInput<T, 2> (0, inputs);
				auto top_diff_real = this->template getInput<T, 2> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 2> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 2> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 2> (1, outputs);

				dim3 dim_block = dim3(16 * 16, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(channels_, dim_block.y));

				maxpooling1d_complex_grad_backward_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, height_out_,
					kernel_h_,
					stride_h_,
					channels_, alpha_, beta_, pad_h_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
			 	//batch_=True

				auto indices = this->template getInput<T, 3> (0, inputs);
				auto top_diff_real = this->template getInput<T, 3> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 3> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 3> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 3> (1, outputs);

				dim3 dim_block = dim3(16 * 16, 4);
				dim3 dim_grid = dim3(divUp(batch_, dim_block.x),
					divUp(height_out_, dim_block.y),
					divUp(channels_, dim_block.z));
				maxpooling1d_complex_grad_backward_with_batch_and_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, height_out_,
					kernel_h_,
					stride_h_,
					channels_, alpha_, beta_, pad_h_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
	}
#define REGISTER_OP(T)\
template class optox::MaxPooling1d_Operator<T> ;
OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP



template < typename T>
	__global__
void maxpooling2d_complex(
            const typename optox::DTensor<T, 3>::ConstRef input_real,
            const typename optox::DTensor<T, 3>::ConstRef input_imag,
			typename optox::DTensor<T, 3>::Ref output_real,
			typename optox::DTensor<T, 3>::Ref output_imag,
			typename optox::DTensor<T, 3>::Ref output_idx,
			int height_in, int width_in,
			int height_out, int width_out,
			int kernel_h, int kernel_w,
			int stride_h, int stride_w,
			int channels, float alpha, float beta,
			int pad_h, int pad_w)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && ch < channels)
	{
		T maxval_real = 0;
		T maxval_imag = 0;

		int maxidx = -1;
		int hstart = i * stride_h -pad_h;
		int wstart = j * stride_w -pad_w;

		int hend = min(hstart + kernel_h, height_in);
		int wend = min(wstart + kernel_w, width_in);

		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				T real = input_real(h, w, ch);
				T imag = input_imag(h, w, ch);

				T amp = alpha *real *real + beta *imag * imag;
				T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
				int idx = h *width_in + w;
				if (amp > max_value)
				{
					maxidx = idx;
					maxval_real = real;
					maxval_imag = imag;
				}
			}
		}

		output_real(i, j, ch) = maxval_real;
		output_imag(i, j, ch) = maxval_imag;
		output_idx(i, j, ch) = maxidx;
	}
}

template < typename T>
	__global__
void maxpooling2d_complex_with_batch(
            const typename optox::DTensor<T, 4>::ConstRef input_real,
            const typename optox::DTensor<T, 4>::ConstRef input_imag,
			typename optox::DTensor<T, 4>::Ref output_real,
			typename optox::DTensor<T, 4>::Ref output_imag,
			typename optox::DTensor<T, 4>::Ref output_idx,
			int height_in, int width_in,
			int height_out, int width_out,
			int kernel_h, int kernel_w,
			int stride_h, int stride_w,
			int channels, float alpha, float beta,
			int pad_h, int pad_w, int batch = 1)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && ch < channels)
	{
		for (int b = 0; b < batch; b++)
		{
			T maxval_real = 0;
			T maxval_imag = 0;

			int maxidx = -1;
			int hstart = i * stride_h -pad_h;
			int wstart = j * stride_w -pad_w;

			int hend = min(hstart + kernel_h, height_in);
			int wend = min(wstart + kernel_w, width_in);

			for (int h = hstart; h < hend; ++h)
			{
				for (int w = wstart; w < wend; ++w)
				{
					T real = input_real(b, h, w, ch);
					T imag = input_imag(b, h, w, ch);

					T amp = alpha *real *real + beta *imag * imag;
					T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
					int idx = h *width_in + w;
					if (amp > max_value)
					{
						maxidx = idx;
						maxval_real = real;
						maxval_imag = imag;
					}
				}
			}

			output_real(b, i, j, ch) = maxval_real;
			output_imag(b, i, j, ch) = maxval_imag;
			output_idx(b, i, j, ch) = maxidx;
		}
	}
}

template < typename T>
	void optox::MaxPooling2d_Operator<T>::computeForward(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs
)
	{
		if (!batch_)
		{
			auto in_real = this->template getInput<T, 3> (0, inputs);
			auto in_imag = this->template getInput<T, 3> (1, inputs);
			auto out_real = this->template getOutput<T, 3> (0, outputs);
			auto out_imag = this->template getOutput<T, 3> (1, outputs);
			auto out_idx = this->template getOutput<T, 3> (2, outputs);
			dim3 dim_block = dim3(16, 16, 4);
			dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
				divUp(width_out_, dim_block.y),
				divUp(channels_, dim_block.z)
		);

			maxpooling2d_complex<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, width_in_, height_out_, width_out_,
				kernel_h_, kernel_w_,
				stride_h_, stride_w_,
				channels_, alpha_, beta_, pad_h_, pad_w_);

			OPTOX_CUDA_CHECK;
		}
		else
		{
			auto in_real = this->template getInput<T, 4> (0, inputs);
			auto in_imag = this->template getInput<T, 4> (1, inputs);
			auto out_real = this->template getOutput<T, 4> (0, outputs);
			auto out_imag = this->template getOutput<T, 4> (1, outputs);
			auto out_idx = this->template getOutput<T, 4> (2, outputs);
			dim3 dim_block = dim3(16, 8, 4);
			dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
				divUp(width_out_, dim_block.y),
				divUp(channels_, dim_block.z)
		);

			maxpooling2d_complex_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, width_in_, height_out_, width_out_,
				kernel_h_, kernel_w_,
				stride_h_, stride_w_,
				channels_, alpha_, beta_, pad_h_, pad_w_, batch_);

			OPTOX_CUDA_CHECK;
		}
	}

template < typename T>
	__global__
void maxpooling2d_complex_grad_backward(
                            const typename optox::DTensor<T, 3>::ConstRef input_real,
                            const typename optox::DTensor<T, 3>::ConstRef input_imag,
                            const typename optox::DTensor<T, 3>::ConstRef output_real,
                            const typename optox::DTensor<T, 3>::ConstRef output_imag,
                            const typename optox::DTensor<T, 3>::ConstRef top_diff_real,
                            const typename optox::DTensor<T, 3>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 3>::Ref bottom_diff_real,
							typename optox::DTensor<T, 3>::Ref bottom_diff_imag,
							int height_in, int width_in,
							int height_out, int width_out,
							int kernel_h, int kernel_w,
							int stride_h, int stride_w,
							int channels, float alpha, float beta,
							int pad_h, int pad_w, int n = 0)
{

	for (int h = 0; h < height_in; h++)
		for (int w = 0; w < width_in; h++)
			for (int ch = 0; ch < channels; ch++)
			{
				bottom_diff_real(h, w, ch) = 0;
				bottom_diff_imag(h, w, ch) = 0;
			}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && ch < channels)
	{
		int maxidx_h = -1;
		int maxidx_w = -1;

		int hstart = i * stride_h -pad_h;
		int wstart = j * stride_w -pad_w;

		int hend = min(hstart + kernel_h, height_in);
		int wend = min(wstart + kernel_w, width_in);
		bool should_stop = false;

		for (int h = hstart; h < hend && !should_stop; ++h)
		{
			for (int w = wstart; w < wend && !should_stop; ++w)
			{
				if (output_real(i, j, ch) == input_real(h, w, ch) && output_imag(i, j, ch) == input_imag(h, w, ch))
				{
					maxidx_h = h;
					maxidx_w = w;
					should_stop = true;
				}
			}
		}

		if (maxidx_h != -1 && maxidx_w != -1)
		{
			bottom_diff_real(maxidx_h, maxidx_w, ch) = top_diff_real(i, j, ch);
			bottom_diff_imag(maxidx_h, maxidx_w, ch) = top_diff_imag(i, j, ch);
		}
	}
}

template < typename T>
	__global__
void maxpooling2d_complex_grad_backward_with_batch(
                            const typename optox::DTensor<T, 4>::ConstRef input_real,
                            const typename optox::DTensor<T, 4>::ConstRef input_imag,
                            const typename optox::DTensor<T, 4>::ConstRef output_real,
                            const typename optox::DTensor<T, 4>::ConstRef output_imag,
                            const typename optox::DTensor<T, 4>::ConstRef top_diff_real,
                            const typename optox::DTensor<T, 4>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 4>::Ref bottom_diff_real,
							typename optox::DTensor<T, 4>::Ref bottom_diff_imag,
							int height_in, int width_in,
							int height_out, int width_out,
							int kernel_h, int kernel_w,
							int stride_h, int stride_w,
							int channels, float alpha, float beta,
							int pad_h, int pad_w, int batch = 1)
{
	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int w = 0; w < width_in; w++)
				for (int ch = 0; ch < channels; ch++)
				{
					bottom_diff_real(b, h, w, ch) = 0;
					bottom_diff_imag(b, h, w, ch) = 0;
				}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;
	if (i < height_out && j < width_out && ch < channels)
	{
		for (int b = 0; b < batch; b++)
		{
			int maxidx_h = -1;
			int maxidx_w = -1;

			int hstart = i * stride_h -pad_h;
			int wstart = j * stride_w -pad_w;

			int hend = min(hstart + kernel_h, height_in);
			int wend = min(wstart + kernel_w, width_in);
			bool should_stop = false;

			for (int h = hstart; h < hend && !should_stop; ++h)
			{
				for (int w = wstart; w < wend && !should_stop; ++w)
				{
					if (output_real(b, i, j, ch) == input_real(b, h, w, ch) && output_imag(b, i, j, ch) == input_imag(b, h, w, ch))
					{
						maxidx_h = h;
						maxidx_w = w;
						should_stop = true;
					}
				}
			}

			if (maxidx_h != -1 && maxidx_w != -1)
			{
				bottom_diff_real(b, maxidx_h, maxidx_w, ch) = top_diff_real(b, i, j, ch);
				bottom_diff_imag(b, maxidx_h, maxidx_w, ch) = top_diff_imag(b, i, j, ch);
			}
		}
	}
}

template < typename T>
	__global__
void maxpooling2d_complex_grad_backward_indices(
                const typename optox::DTensor<T, 3>::ConstRef indices,
                const typename optox::DTensor<T, 3>::ConstRef top_diff_real,
                const typename optox::DTensor<T, 3>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 3>::Ref bottom_diff_real,
				typename optox::DTensor<T, 3>::Ref bottom_diff_imag,
				int height_in, int width_in,
				int height_out, int width_out,
				int kernel_h, int kernel_w,
				int stride_h, int stride_w,
				int channels, float alpha, float beta,
				int pad_h, int pad_w)
{

	for (int h = 0; h < height_in; h++)
		for (int w = 0; w < width_in; h++)
			for (int ch = 0; ch < channels; ch++)
			{
				bottom_diff_real(h, w, ch) = 0;
				bottom_diff_imag(h, w, ch) = 0;
			}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && ch < channels)
	{
		int idx = indices(i, j, ch);
		int n = idx;
		int w = n % width_in;
		n /= width_in;
		int h = n % height_in;
		bottom_diff_real(h, w, ch) = top_diff_real(i, j, ch);
		bottom_diff_imag(h, w, ch) = top_diff_imag(i, j, ch);
	}
}

template < typename T>
	__global__
void maxpooling2d_complex_grad_backward_with_batch_and_indices(
                const typename optox::DTensor<T, 4>::ConstRef indices,
                const typename optox::DTensor<T, 4>::ConstRef top_diff_real,
                const typename optox::DTensor<T, 4>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 4>::Ref bottom_diff_real,
				typename optox::DTensor<T, 4>::Ref bottom_diff_imag,
				int height_in, int width_in,
				int height_out, int width_out,
				int kernel_h, int kernel_w,
				int stride_h, int stride_w,
				int channels, float alpha, float beta,
				int pad_h, int pad_w, int batch = 1)
{

	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int w = 0; w < width_in; w++)
				for (int ch = 0; ch < channels; ch++)
				{
					bottom_diff_real(b, h, w, ch) = 0;
					bottom_diff_imag(b, h, w, ch) = 0;
				}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int ch = threadIdx.z + blockIdx.z *blockDim.z;
	if (i < height_out && j < width_out && ch < channels)
	{
		for (int b = 0; b < batch; b++)
		{
			int idx = indices(b, i, j, ch);
			int n = idx;
			int w = n % width_in;
			n /= width_in;
			int h = n % height_in;
			bottom_diff_real(b, h, w, ch) = top_diff_real(b, i, j, ch);
			bottom_diff_imag(b, h, w, ch) = top_diff_imag(b, i, j, ch);
		}
	}
}

template < typename T>
	void optox::MaxPooling2d_Operator<T>::computeAdjoint(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs

)
	{
		if (!with_indices_)
		{
			if (!batch_)
			{
				auto in_real = this->template getInput<T, 3> (0, inputs);
				auto in_imag = this->template getInput<T, 3> (1, inputs);

				auto out_real = this->template getInput<T, 3> (2, inputs);
				auto out_imag = this->template getInput<T, 3> (3, inputs);

				auto top_diff_real = this->template getInput<T, 3> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 3> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 3> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 3> (1, outputs);

				dim3 dim_block = dim3(16, 16, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(channels_, dim_block.z));

				maxpooling2d_complex_grad_backward<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, height_out_, width_out_,
					kernel_h_, kernel_w_,
					stride_h_, stride_w_,
					channels_, alpha_, beta_, pad_h_, pad_w_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				//batch=true
				auto in_real = this->template getInput<T, 4> (0, inputs);
				auto in_imag = this->template getInput<T, 4> (1, inputs);

				auto out_real = this->template getInput<T, 4> (2, inputs);
				auto out_imag = this->template getInput<T, 4> (3, inputs);

				auto top_diff_real = this->template getInput<T, 4> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 4> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 4> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 4> (1, outputs);

				dim3 dim_block = dim3(16, 8, 2);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(channels_, dim_block.z));

				maxpooling2d_complex_grad_backward_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, height_out_, width_out_,
					kernel_h_, kernel_w_,
					stride_h_, stride_w_,
					channels_, alpha_, beta_, pad_h_, pad_w_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
		else
		{
			//indices
			if (!batch_)
			{
				auto indices = this->template getInput<T, 3> (0, inputs);
				auto top_diff_real = this->template getInput<T, 3> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 3> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 3> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 3> (1, outputs);

				dim3 dim_block = dim3(16, 4, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(channels_, dim_block.z));

				maxpooling2d_complex_grad_backward_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, height_out_, width_out_,
					kernel_h_, kernel_w_,
					stride_h_, stride_w_,
					channels_, alpha_, beta_, pad_h_, pad_w_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				//batch_=True

				auto indices = this->template getInput<T, 4> (0, inputs);
				auto top_diff_real = this->template getInput<T, 4> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 4> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 4> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 4> (1, outputs);

				dim3 dim_block = dim3(16, 4, 2);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(channels_, dim_block.z));
				maxpooling2d_complex_grad_backward_with_batch_and_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, height_out_, width_out_,
					kernel_h_, kernel_w_,
					stride_h_, stride_w_,
					channels_, alpha_, beta_, pad_h_, pad_w_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
	}
#define REGISTER_OP(T)\
template class optox::MaxPooling2d_Operator<T> ;
OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP

template < typename T>
	__global__
void maxpooling3d_complex(
            const typename optox::DTensor<T, 4>::ConstRef input_real,
            const typename optox::DTensor<T, 4>::ConstRef input_imag,
			typename optox::DTensor<T, 4>::Ref output_real,
			typename optox::DTensor<T, 4>::Ref output_imag,
			typename optox::DTensor<T, 4>::Ref output_idx,
			int height_in, int width_in, int depth_in,
			int height_out, int width_out, int depth_out,
			int kernel_h, int kernel_w, int kernel_d,
			int stride_h, int stride_w, int stride_d,
			int channels, float alpha, float beta,
			int pad_h, int pad_w, int pad_d)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int ch = 0; ch < channels; ch++)
		{
			T maxval_real = 0;
			T maxval_imag = 0;

			int maxidx = -1;
			int hstart = i *stride_h - pad_h;
			int wstart = j *stride_w - pad_w;
			int dstart = k *stride_d - pad_d;

			int hend = min(hstart + kernel_h, height_in);
			int wend = min(wstart + kernel_w, width_in);
			int dend = min(dstart + kernel_d, depth_in);

			for (int h = hstart; h < hend; ++h)
			{
				for (int w = wstart; w < wend; ++w)
				{
					for (int d = dstart; d < dend; ++d)
					{
						T real = input_real(h, w, d, ch);
						T imag = input_imag(h, w, d, ch);

						T amp = alpha *real *real + beta *imag * imag;
						T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
						int idx = (h *width_in + w) *depth_in + d;

						if (amp > max_value)
						{
							maxidx = idx;
							maxval_real = real;
							maxval_imag = imag;
						}
					}
				}
			}

			output_real(i, j, k, ch) = maxval_real;
			output_imag(i, j, k, ch) = maxval_imag;
			output_idx(i, j, k, ch) = maxidx;
		}
	}
}

template < typename T>
	__global__
void maxpooling3d_complex_with_batch(
	        const typename optox::DTensor<T, 5>::ConstRef input_real,
	        const typename optox::DTensor<T, 5>::ConstRef input_imag,
			typename optox::DTensor<T, 5>::Ref output_real,
			typename optox::DTensor<T, 5>::Ref output_imag,
			typename optox::DTensor<T, 5>::Ref output_idx,
			int height_in, int width_in, int depth_in,
			int height_out, int width_out, int depth_out,
			int kernel_h, int kernel_w, int kernel_d,
			int stride_h, int stride_w, int stride_d,
			int channels, float alpha, float beta,
			int pad_h, int pad_w, int pad_d, int batch = 1)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int b = 0; b < batch; b++)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				T maxval_real = 0;
				T maxval_imag = 0;

				int maxidx = -1;
				int hstart = i *stride_h - pad_h;
				int wstart = j *stride_w - pad_w;
				int dstart = k *stride_d - pad_d;

				int hend = min(hstart + kernel_h, height_in);
				int wend = min(wstart + kernel_w, width_in);
				int dend = min(dstart + kernel_d, depth_in);

				for (int h = hstart; h < hend; ++h)
				{
					for (int w = wstart; w < wend; ++w)
					{
						for (int d = dstart; d < dend; ++d)
						{
							T real = input_real(b, h, w, d, ch);
							T imag = input_imag(b, h, w, d, ch);

							T amp = alpha *real *real + beta *imag * imag;
							T max_value = alpha *maxval_real *maxval_real + beta *maxval_imag * maxval_imag;
							int idx = (h *width_in + w) *depth_in + d;

							if (amp > max_value)
							{
								maxidx = idx;
								maxval_real = real;
								maxval_imag = imag;
							}
						}
					}
				}

				output_real(b, i, j, k, ch) = maxval_real;
				output_imag(b, i, j, k, ch) = maxval_imag;
				output_idx(b, i, j, k, ch) = maxidx;
			}
		}
	}
}

template < typename T>
	void optox::MaxPooling3d_Operator<T>::computeForward(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs

)
	{
		if (!batch_)
		{
			auto in_real = this->template getInput<T, 4> (0, inputs);
			auto in_imag = this->template getInput<T, 4> (1, inputs);
			auto out_real = this->template getOutput<T, 4> (0, outputs);
			auto out_imag = this->template getOutput<T, 4> (1, outputs);
			auto out_idx = this->template getOutput<T, 4> (2, outputs);

			dim3 dim_block = dim3(16, 16, 4);
			dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
				divUp(width_out_, dim_block.y),
				divUp(depth_out_, dim_block.z)
		);

			maxpooling3d_complex<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, width_in_, depth_in_, height_out_, width_out_, depth_out_,
				kernel_h_, kernel_w_, kernel_d_,
				stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, pad_h_, pad_w_, pad_d_);

			OPTOX_CUDA_CHECK;
		}
		else
		{
			auto in_real = this->template getInput<T, 5> (0, inputs);
			auto in_imag = this->template getInput<T, 5> (1, inputs);
			auto out_real = this->template getOutput<T, 5> (0, outputs);
			auto out_imag = this->template getOutput<T, 5> (1, outputs);
			auto out_idx = this->template getOutput<T, 5> (2, outputs);
			dim3 dim_block = dim3(16, 8, 2);
			dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
				divUp(width_out_, dim_block.y),
				divUp(depth_out_, dim_block.z)
		);

			maxpooling3d_complex_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, height_in_, width_in_, depth_in_, height_out_, width_out_, depth_out_,
				kernel_h_, kernel_w_, kernel_d_,
				stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, pad_h_, pad_w_, pad_d_, batch_);

			OPTOX_CUDA_CHECK;
		}
	}

template < typename T>
	__global__
void maxpooling3d_complex_grad_backward(
                            const typename optox::DTensor<T, 4>::ConstRef input_real,
                            const typename optox::DTensor<T, 4>::ConstRef input_imag,
                            const typename optox::DTensor<T, 4>::ConstRef output_real,
                            const typename optox::DTensor<T, 4>::ConstRef output_imag,
                            const typename optox::DTensor<T, 4>::ConstRef top_diff_real,
                            const typename optox::DTensor<T, 4>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 4>::Ref bottom_diff_real,
							typename optox::DTensor<T, 4>::Ref bottom_diff_imag,
							int height_in, int width_in, int depth_in,
							int height_out, int width_out, int depth_out,
							int kernel_h, int kernel_w, int kernel_d,
							int stride_h, int stride_w, int stride_d,
							int channels, float alpha, float beta,
							int pad_h, int pad_w, int pad_d, int n = 0)
{
	for (int h = 0; h < height_in; h++)
		for (int w = 0; w < width_in; w++)
			for (int d = 0; d < depth_in; d++)
				for (int ch = 0; ch < channels; ch++)
				{
					bottom_diff_real(h, w, d, ch) = 0;
					bottom_diff_imag(h, w, d, ch) = 0;
				}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int ch = 0; ch < channels; ch++)
		{
			int maxidx_h = -1;
			int maxidx_w = -1;
			int maxidx_d = -1;

			int hstart = i *stride_h - pad_h;
			int wstart = j *stride_w - pad_w;
			int dstart = k *stride_d - pad_d;

			int hend = min(hstart + kernel_h, height_in);
			int wend = min(wstart + kernel_w, width_in);
			int dend = min(dstart + kernel_d, depth_in);
			bool should_stop = false;

			for (int h = hstart; h < hend && !should_stop; ++h)
			{
				for (int w = wstart; w < wend && !should_stop; ++w)
				{
					for (int d = dstart; d < dend && !should_stop; ++d)
					{
						if (output_real(i, j, k, ch) == input_real(h, w, d, ch) && output_imag(i, j, k, ch) == input_imag(h, w, d, ch))
						{
							maxidx_h = h;
							maxidx_w = w;
							maxidx_d = d;
							should_stop = true;
						}
					}
				}
			}

			if (maxidx_h != -1 && maxidx_w != -1 && maxidx_d != -1)
			{
				bottom_diff_real(maxidx_h, maxidx_w, maxidx_d, ch) = top_diff_real(i, j, k, ch);
				bottom_diff_imag(maxidx_h, maxidx_w, maxidx_d, ch) = top_diff_imag(i, j, k, ch);
			}
		}
	}
}

template < typename T>
	__global__
void maxpooling3d_complex_grad_backward_with_batch(
                            const typename optox::DTensor<T, 5>::ConstRef input_real,
                            const typename optox::DTensor<T, 5>::ConstRef input_imag,
                            const typename optox::DTensor<T, 5>::ConstRef output_real,
                            const typename optox::DTensor<T, 5>::ConstRef output_imag,
                            const typename optox::DTensor<T, 5>::ConstRef top_diff_real,
                            const typename optox::DTensor<T, 5>::ConstRef top_diff_imag,
							typename optox::DTensor<T, 5>::Ref bottom_diff_real,
							typename optox::DTensor<T, 5>::Ref bottom_diff_imag,
							int height_in, int width_in, int depth_in,
							int height_out, int width_out, int depth_out,
							int kernel_h, int kernel_w, int kernel_d,
							int stride_h, int stride_w, int stride_d,
							int channels, float alpha, float beta,
							int pad_h, int pad_w, int pad_d, int batch = 1)
{
	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int w = 0; w < width_in; w++)
				for (int d = 0; d < depth_in; d++)
					for (int ch = 0; ch < channels; ch++)
					{
						bottom_diff_real(b, h, w, d, ch) = 0;
						bottom_diff_imag(b, h, w, d, ch) = 0;
					}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int b = 0; b < batch; b++)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				int maxidx_h = -1;
				int maxidx_w = -1;
				int maxidx_d = -1;

				int hstart = i *stride_h - pad_h;
				int wstart = j *stride_w - pad_w;
				int dstart = k *stride_d - pad_d;

				int hend = min(hstart + kernel_h, height_in);
				int wend = min(wstart + kernel_w, width_in);
				int dend = min(dstart + kernel_d, depth_in);
				bool should_stop = false;

				for (int h = hstart; h < hend && !should_stop; ++h)
				{
					for (int w = wstart; w < wend && !should_stop; ++w)
					{
						for (int d = dstart; d < dend && !should_stop; ++d)
						{
							if (output_real(b, i, j, k, ch) == input_real(b, h, w, d, ch) && output_imag(b, i, j, k, ch) == input_imag(b, h, w, d, ch))
							{
								maxidx_h = h;
								maxidx_w = w;
								maxidx_d = d;
								should_stop = true;
							}
						}
					}
				}

				if (maxidx_h != -1 && maxidx_w != -1 && maxidx_d != -1)
				{
					bottom_diff_real(b, maxidx_h, maxidx_w, maxidx_d, ch) = top_diff_real(b, i, j, k, ch);
					bottom_diff_imag(b, maxidx_h, maxidx_w, maxidx_d, ch) = top_diff_imag(b, i, j, k, ch);
				}
			}
		}
	}
}

template < typename T>
	__global__
void maxpooling3d_complex_grad_backward_indices(
                const typename optox::DTensor<T, 4>::ConstRef indices,
                const typename optox::DTensor<T, 4>::ConstRef top_diff_real,
                const typename optox::DTensor<T, 4>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 4>::Ref bottom_diff_real,
				typename optox::DTensor<T, 4>::Ref bottom_diff_imag,
				int height_in, int width_in, int depth_in,
				int height_out, int width_out, int depth_out,
				int kernel_h, int kernel_w, int kernel_d,
				int stride_h, int stride_w, int stride_d,
				int channels, float alpha, float beta,
				int pad_h, int pad_w, int pad_d)
{

	for (int h = 0; h < height_in; h++)
		for (int w = 0; w < width_in; w++)
			for (int d = 0; d < depth_in; d++)
				for (int ch = 0; ch < channels; ch++)
				{
					bottom_diff_real(h, w, d, ch) = 0;
					bottom_diff_imag(h, w, d, ch) = 0;
				}

	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int ch = 0; ch < channels; ch++)
		{
			int idx = indices(i, j, k, ch);
			int n = idx;
			int d = idx % depth_in;
			n /= depth_in;
			int w = n % width_in;
			n /= width_in;
			int h = n % height_in;
			bottom_diff_real(h, w, d, ch) = top_diff_real(i, j, k, ch);
			bottom_diff_imag(h, w, d, ch) = top_diff_imag(i, j, k, ch);
		}
	}
}

template < typename T>
	__global__
void maxpooling3d_complex_grad_backward_with_batch_and_indices(	const typename optox::DTensor<T, 5>::ConstRef indices, const typename optox::DTensor<T, 5>::ConstRef top_diff_real, const typename optox::DTensor<T, 5>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 5>::Ref bottom_diff_real,
				typename optox::DTensor<T, 5>::Ref bottom_diff_imag,
				int height_in, int width_in, int depth_in,
				int height_out, int width_out, int depth_out,
				int kernel_h, int kernel_w, int kernel_d,
				int stride_h, int stride_w, int stride_d,
				int channels, float alpha, float beta,
				int pad_h, int pad_w, int pad_d, int batch = 1)
{
	for (int b = 0; b < batch; b++)
		for (int h = 0; h < height_in; h++)
			for (int w = 0; w < width_in; w++)
				for (int d = 0; d < depth_in; d++)
					for (int ch = 0; ch < channels; ch++)
					{
						bottom_diff_real(b, h, w, d, ch) = 0;
						bottom_diff_imag(b, h, w, d, ch) = 0;
					}
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	int j = threadIdx.y + blockIdx.y *blockDim.y;
	int k = threadIdx.z + blockIdx.z *blockDim.z;

	if (i < height_out && j < width_out && k < depth_out)
	{
		for (int b = 0; b < batch; b++)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				int idx = indices(b, i, j, k, ch);
				int n = idx;
				int d = idx % depth_in;
				n /= depth_in;
				int w = n % width_in;
				n /= width_in;
				int h = n % height_in;
				bottom_diff_real(b, h, w, d, ch) = top_diff_real(b, i, j, k, ch);
				bottom_diff_imag(b, h, w, d, ch) = top_diff_imag(b, i, j, k, ch);
			}
		}
	}
}

template < typename T>
	void optox::MaxPooling3d_Operator<T>::computeAdjoint(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs

)
	{
		if (!with_indices_)
		{
			if (!batch_)
			{
				auto in_real = this->template getInput<T, 4> (0, inputs);
				auto in_imag = this->template getInput<T, 4> (1, inputs);

				auto out_real = this->template getInput<T, 4> (2, inputs);
				auto out_imag = this->template getInput<T, 4> (3, inputs);

				auto top_diff_real = this->template getInput<T, 4> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 4> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 4> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 4> (1, outputs);

				dim3 dim_block = dim3(16, 16, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(depth_out_, dim_block.z));

				maxpooling3d_complex_grad_backward<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, depth_in_, height_out_, width_out_, depth_out_,
					kernel_h_, kernel_w_, kernel_d_,
					stride_h_, stride_w_, stride_d_,
					channels_, alpha_, beta_, pad_h_, pad_w_, pad_d_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				auto in_real = this->template getInput<T, 5> (0, inputs);
				auto in_imag = this->template getInput<T, 5> (1, inputs);

				auto out_real = this->template getInput<T, 5> (2, inputs);
				auto out_imag = this->template getInput<T, 5> (3, inputs);

				auto top_diff_real = this->template getInput<T, 5> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 5> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 5> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 5> (1, outputs);

				dim3 dim_block = dim3(16, 4, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(depth_out_, dim_block.z));

				maxpooling3d_complex_grad_backward_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, depth_in_, height_out_, width_out_, depth_out_,
					kernel_h_, kernel_w_, kernel_d_,
					stride_h_, stride_w_, stride_d_,
					channels_, alpha_, beta_, pad_h_, pad_w_, pad_d_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
		else
		{
			if (!batch_)
			{
				auto indices = this->template getInput<T, 4> (0, inputs);
				auto top_diff_real = this->template getInput<T, 4> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 4> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 4> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 4> (1, outputs);

				dim3 dim_block = dim3(16, 4, 4);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(depth_out_, dim_block.z));

				maxpooling3d_complex_grad_backward_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, depth_in_, height_out_, width_out_, depth_out_,
					kernel_h_, kernel_w_, kernel_d_,
					stride_h_, stride_w_, stride_d_,
					channels_, alpha_, beta_, pad_h_, pad_w_, pad_d_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				//batch_=True
				auto indices = this->template getInput<T, 5> (0, inputs);
				auto top_diff_real = this->template getInput<T, 5> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 5> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 5> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 5> (1, outputs);

				dim3 dim_block = dim3(16, 4, 2);
				dim3 dim_grid = dim3(divUp(height_out_, dim_block.x),
					divUp(width_out_, dim_block.y),
					divUp(depth_out_, dim_block.z));

				maxpooling3d_complex_grad_backward_with_batch_and_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
					height_in_, width_in_, depth_in_,
					height_out_, width_out_, depth_out_,
					kernel_h_, kernel_w_, kernel_d_,
					stride_h_, stride_w_, stride_d_,
					channels_, alpha_, beta_,
					pad_h_, pad_w_, pad_d_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
	}
#define REGISTER_OP(T)\
template class optox::MaxPooling3d_Operator<T> ;
OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP

template < typename T >
  __global__
void maxpooling4d_complex(
  const typename optox::DTensor < T, 1 > ::ConstRef input_real,
    const typename optox::DTensor < T, 1 > ::ConstRef input_imag,
      typename optox::DTensor < T, 1 > ::Ref output_real,
      typename optox::DTensor < T, 1 > ::Ref output_imag,
      typename optox::DTensor < T, 1 > ::Ref output_idx,

      int time_in, int height_in, int width_in, int depth_in,
      int time_out, int height_out, int width_out, int depth_out,
      int kernel_t, int kernel_h, int kernel_w, int kernel_d,
      int stride_t, int stride_h, int stride_w, int stride_d,
      int channels, float alpha, float beta,
      int pad_t, int pad_h, int pad_w, int pad_d) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < time_out * height_out * width_out * depth_out * channels)

  {
    int n = index;
    int ch = n % channels;
    n /= channels;
    int k = n % depth_out;
    n /= depth_out;
    int j = n % width_out;
    n /= width_out;
    int i = n % height_out;
    n /= height_out;
    int ti = n % time_out;

    T maxval_real = 0;
    T maxval_imag = 0;

    int maxidx = -1;
    int tstart = ti * stride_t - pad_t;
    int hstart = i * stride_h - pad_h;
    int wstart = j * stride_w - pad_w;
    int dstart = k * stride_d- pad_d;

    int tend = min(tstart + kernel_t, time_in);
    int hend = min(hstart + kernel_h, height_in);
    int wend = min(wstart + kernel_w, width_in);
    int dend = min(dstart + kernel_d, depth_in);

    for (int t = tstart; t < tend; ++t) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          for (int d = dstart; d < dend; ++d)

          {
            int idx = t * height_in * width_in * depth_in * channels + h * width_in * depth_in * channels + w * depth_in * channels + d * channels + ch;
            T real = input_real(idx);
            T imag = input_imag(idx);

            T amp = alpha * real * real + beta * imag * imag;
            T max_value = alpha * maxval_real * maxval_real + beta * maxval_imag * maxval_imag;
            //int idx = h *width_in + w;

            if (amp > max_value) {
              maxidx = idx;
              maxval_real = real;
              maxval_imag = imag;
            }
          }
        }
      }
    }
    int out_idx = ti * height_out * width_out * depth_out * channels + i * width_out * depth_out * channels + j * depth_out * channels + k * channels + ch;
    output_real(out_idx) = maxval_real;
    output_imag(out_idx) = maxval_imag;
    output_idx(out_idx) = maxidx;
  }
}


template < typename T >
  __global__
void maxpooling4d_complex_with_batch(
  const typename optox::DTensor < T, 1 > ::ConstRef input_real,
    const typename optox::DTensor < T, 1 > ::ConstRef input_imag,
      typename optox::DTensor < T, 1 > ::Ref output_real,
      typename optox::DTensor < T, 1 > ::Ref output_imag,
      typename optox::DTensor < T, 1 > ::Ref output_idx,

      int time_in, int height_in, int width_in, int depth_in,
      int time_out, int height_out, int width_out, int depth_out,
      int kernel_t, int kernel_h, int kernel_w, int kernel_d,
      int stride_t, int stride_h, int stride_w, int stride_d,
      int channels, float alpha, float beta,
      int pad_t, int pad_h, int pad_w, int pad_d, int batch = 1) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < batch * time_out * height_out * width_out * depth_out * channels)

  {
    int n = index;
    int ch = n % channels;
    n /= channels;
    int k = n % depth_out;
    n /= depth_out;
    int j = n % width_out;
    n /= width_out;
    int i = n % height_out;
    n /= height_out;
    int ti = n % time_out;
    n /= time_out;
    int b = n % batch;

    
    

    T maxval_real = 0;
    T maxval_imag = 0;

    int maxidx = -1;
    int tstart = ti * stride_t - pad_t;
    int hstart = i * stride_h - pad_h;
    int wstart = j * stride_w - pad_w;
    int dstart = k * stride_d- pad_d;


    int tend = min(tstart + kernel_t, time_in);
    int hend = min(hstart + kernel_h, height_in);
    int wend = min(wstart + kernel_w, width_in);
    int dend = min(dstart + kernel_d, depth_in);
    for (int t = tstart; t < tend; ++t) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          for (int d = dstart; d < dend; ++d)

          {
            int idx = b * time_in *  height_in * width_in * depth_in * channels + t * height_in * width_in * depth_in * channels + h * width_in * depth_in * channels + w * depth_in * channels + d * channels + ch;
            T real = input_real(idx);
            T imag = input_imag(idx);

            T amp = alpha * real * real + beta * imag * imag;
            T max_value = alpha * maxval_real * maxval_real + beta * maxval_imag * maxval_imag;
            //int idx = h *width_in + w;

            if (amp > max_value) {
              maxidx = idx;
              maxval_real = real;
              maxval_imag = imag;
            }
          }
        }
      }
    }
    int out_idx = b* time_out* height_out * width_out * depth_out * channels+ ti * height_out * width_out * depth_out * channels + i * width_out * depth_out * channels + j * depth_out * channels + k * channels + ch;
    output_real(out_idx) = maxval_real;
    output_imag(out_idx) = maxval_imag;
    output_idx(out_idx) = maxidx;
  }
}




template < typename T>
	void optox::MaxPooling4d_Operator<T>::computeForward(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs
)
	{
		if (!batch_)
		{
			auto in_real = this->template getInput<T, 1> (0, inputs);
			auto in_imag = this->template getInput<T, 1> (1, inputs);
			auto out_real = this->template getOutput<T, 1> (0, outputs);
			auto out_imag = this->template getOutput<T, 1> (1, outputs);
			auto out_idx = this->template getOutput<T, 1> (2, outputs);
			dim3 dim_block = dim3(16 * 16 * 4); 
			dim3 dim_grid = dim3(divUp(time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));



			maxpooling4d_complex<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx, 
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_);

			OPTOX_CUDA_CHECK;
		}
		else
		{
			auto in_real = this->template getInput<T, 1> (0, inputs);
			auto in_imag = this->template getInput<T, 1> (1, inputs);
			auto out_real = this->template getOutput<T, 1> (0, outputs);
			auto out_imag = this->template getOutput<T, 1> (1, outputs);
			auto out_idx = this->template getOutput<T, 1> (2, outputs);
			dim3 dim_block = dim3(16 * 16 * 4); 
			dim3 dim_grid = dim3(divUp(batch_*time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));


			maxpooling4d_complex_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
				*out_real, *out_imag, *out_idx,				
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_, batch_);

			OPTOX_CUDA_CHECK;
		}
	}
	





template < typename T >
  __global__
void maxpooling4d_complex_grad_backward(
  	      const typename optox::DTensor < T, 1 > ::ConstRef input_real,
    	      const typename optox::DTensor < T, 1 > ::ConstRef input_imag,
              const typename optox::DTensor < T, 1 > ::ConstRef output_real,
              const typename optox::DTensor < T, 1 > ::ConstRef output_imag,
              const typename optox::DTensor < T, 1 > ::ConstRef top_diff_real,
              const typename optox::DTensor < T, 1 > ::ConstRef top_diff_imag,
              typename optox::DTensor < T, 1 > ::Ref bottom_diff_real,
              typename optox::DTensor < T, 1 > ::Ref bottom_diff_imag,
              int time_in, int height_in, int width_in, int depth_in,
              int time_out, int height_out, int width_out, int depth_out,
              int kernel_t, int kernel_h, int kernel_w, int kernel_d,
              int stride_t, int stride_h, int stride_w, int stride_d,
              int channels, float alpha, float beta,
              int pad_t, int pad_h, int pad_w, int pad_d) {
  for (int n = 0; n < time_in * height_in * width_in * channels; n++)
  {
    bottom_diff_real(n) = 0;
    bottom_diff_imag(n) = 0;
  }
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < time_out * height_out * width_out * depth_out * channels) {
    int n = index;
    int ch = n % channels;
    n /= channels;
    int k = n % depth_out;
    n /= depth_out;
    int j = n % width_out;
    n /= width_out;
    int i = n % height_out;
    n /= height_out;
    int ti = n % time_out;

    int maxidx_t = -1;
    int maxidx_h = -1;
    int maxidx_w = -1;
    int maxidx_d = -1;

    int tstart = ti * stride_t - pad_t;
    int hstart = i * stride_h - pad_h;
    int wstart = j * stride_w - pad_w;
    int dstart = k * stride_d - pad_d;

    int tend = min(tstart + kernel_t, time_in);
    int hend = min(hstart + kernel_h, height_in);
    int wend = min(wstart + kernel_w, width_in);
    int dend = min(dstart + kernel_d, depth_in);
    bool should_stop = false;

    int t_coe = height_out * width_out * depth_out * channels;
    int i_coe = width_out * depth_out * channels;
    int j_coe = depth_out * channels;

    int t_in_coe = height_in * width_in * depth_in * channels;
    int h_in_coe = width_in * depth_in * channels;
    int w_in_coe = depth_in * channels;

    int out_idx =  ti * t_coe + i * i_coe + j * j_coe + k * channels + ch;
    for (int t = hstart; t < hend && !should_stop; ++t) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          for (int d = dstart; d < dend && !should_stop; ++d) {
            int window_idx = t * t_in_coe + h * h_in_coe + w * w_in_coe + d * channels + ch;
            if (output_real(out_idx) == input_real(window_idx) && output_imag(out_idx) == input_imag(window_idx))

            {
              maxidx_t = t;
              maxidx_h = h;
              maxidx_w = w;
              maxidx_d = d;
              should_stop = true;
            }
          }
        }
      }
    }

    if (maxidx_h != -1 && maxidx_w != -1 && maxidx_d != -1) {
      int bottom_idx =  maxidx_t * t_in_coe + maxidx_h * h_in_coe + maxidx_w * w_in_coe + maxidx_d * channels + ch;
      bottom_diff_real(bottom_idx) = top_diff_real(out_idx);
      bottom_diff_imag(bottom_idx) = top_diff_imag(out_idx);
    }
  }
}




template < typename T >
  __global__
void maxpooling4d_complex_grad_backward_with_batch(
  	      const typename optox::DTensor < T, 1 > ::ConstRef input_real,
    	      const typename optox::DTensor < T, 1 > ::ConstRef input_imag,
              const typename optox::DTensor < T, 1 > ::ConstRef output_real,
              const typename optox::DTensor < T, 1 > ::ConstRef output_imag,
              const typename optox::DTensor < T, 1 > ::ConstRef top_diff_real,
              const typename optox::DTensor < T, 1 > ::ConstRef top_diff_imag,
              typename optox::DTensor < T, 1 > ::Ref bottom_diff_real,
              typename optox::DTensor < T, 1 > ::Ref bottom_diff_imag,
              int time_in, int height_in, int width_in, int depth_in,
              int time_out, int height_out, int width_out, int depth_out,
              int kernel_t, int kernel_h, int kernel_w, int kernel_d,
              int stride_t, int stride_h, int stride_w, int stride_d,
              int channels, float alpha, float beta,
              int pad_t, int pad_h, int pad_w, int pad_d, int batch = 1) {
  for (int n = 0; n < batch * time_in * height_in * width_in * channels; n++)

  {
    bottom_diff_real(n) = 0;
    bottom_diff_imag(n) = 0;
  }
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < batch * time_out * height_out * width_out * depth_out * channels) {
    int n = index;
    int ch = n % channels;
    n /= channels;
    int k = n % depth_out;
    n /= depth_out;
    int j = n % width_out;
    n /= width_out;
    int i = n % height_out;
    n /= height_out;
    int ti = n % time_out;
    n /= time_out;
    int b = n % batch;
    int maxidx_t = -1;
    int maxidx_h = -1;
    int maxidx_w = -1;
    int maxidx_d = -1;

    int tstart = ti * stride_t - pad_t;
    int hstart = i * stride_h - pad_h;
    int wstart = j * stride_w - pad_w;
    int dstart = k * stride_d - pad_d;

    int tend = min(tstart + kernel_t, time_in);
    int hend = min(hstart + kernel_h, height_in);
    int wend = min(wstart + kernel_w, width_in);
    int dend = min(dstart + kernel_d, depth_in);
    bool should_stop = false;

    int b_coe = time_out * height_out * width_out * depth_out * channels;
    int t_coe = height_out * width_out * depth_out * channels;
    int i_coe = width_out * depth_out * channels;
    int j_coe = depth_out * channels;

    int b_in_coe = time_in * height_in * width_in * depth_in * channels;
    int t_in_coe = height_in * width_in * depth_in * channels;
    int h_in_coe = width_in * depth_in * channels;
    int w_in_coe = depth_in * channels;

    int out_idx = b * b_coe + ti * t_coe + i * i_coe + j * j_coe + k * channels + ch;
    for (int t = hstart; t < hend && !should_stop; ++t) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          for (int d = dstart; d < dend && !should_stop; ++d) {

            int window_idx = b * b_in_coe + t * t_in_coe + h * h_in_coe + w * w_in_coe + d * channels + ch;

            if (output_real(out_idx) == input_real(window_idx) && output_imag(out_idx) == input_imag(window_idx))

            {
              maxidx_t = t;
              maxidx_h = h;
              maxidx_w = w;
              maxidx_d = d;
              should_stop = true;
            }
          }
        }
      }
    }

    if (maxidx_h != -1 && maxidx_w != -1 && maxidx_d != -1) {
      int bottom_idx = b * b_in_coe + maxidx_t * t_in_coe + maxidx_h * h_in_coe + maxidx_w * w_in_coe + maxidx_d * channels + ch;
      bottom_diff_real(bottom_idx) = top_diff_real(out_idx);
      bottom_diff_imag(bottom_idx) = top_diff_imag(out_idx);
    }

  }
}

template < typename T>
	__global__
void maxpooling4d_complex_grad_backward_with_indices(	
const typename optox::DTensor<T, 1>::ConstRef indices, 
const typename optox::DTensor<T, 1>::ConstRef top_diff_real, 
const typename optox::DTensor<T, 1>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 1>::Ref bottom_diff_real,
				typename optox::DTensor<T, 1>::Ref bottom_diff_imag,
              int time_in, int height_in, int width_in, int depth_in,
              int time_out, int height_out, int width_out, int depth_out,
              int kernel_t, int kernel_h, int kernel_w, int kernel_d,
              int stride_t, int stride_h, int stride_w, int stride_d,
              int channels, float alpha, float beta,
              int pad_t, int pad_h, int pad_w, int pad_d)
{
	for (int n = 0; n < time_in *height_in *width_in * channels; n++)

	{
		bottom_diff_real(n) = 0;
		bottom_diff_imag(n) = 0;
	}

int index = threadIdx.x + blockIdx.x *blockDim.x;

if (index < time_out *height_out *width_out *depth_out *channels)
{
	int n = index;
	int ch = n % channels;
	n /= channels;
	int k = n % depth_out;
	n /= depth_out;
	int j = n % width_out;
	n /= width_out;
	int i = n % height_out;
	n /= height_out;
	int ti = n % time_out;


	int t_coe = height_out *width_out *depth_out * channels;
	int i_coe = width_out *depth_out * channels;
	int j_coe = depth_out * channels;


	int t_in_coe = height_in *width_in *depth_in * channels;
	int h_in_coe = width_in *depth_in * channels;
	int w_in_coe = depth_in * channels;

	int out_idx =  ti *t_coe + i *i_coe + j *j_coe + k *channels + ch;

	int idx = indices(out_idx);

	n = idx;
	int ch_in = n % channels;
	n /= channels;
	int d = idx % depth_in;
	n /= depth_in;
	int w = n % width_in;
	n /= width_in;
	int h = n % height_in;
	n /= height_in;
	int t = n % time_in;


	int bottom_idx =  t *t_in_coe + h * h_in_coe + w *w_in_coe + d *channels + ch_in;

	bottom_diff_real(bottom_idx) = top_diff_real(out_idx);
	bottom_diff_imag(bottom_idx) = top_diff_imag(out_idx);
}
}





template < typename T>
	__global__
void maxpooling4d_complex_grad_backward_with_batch_and_indices(
const typename optox::DTensor<T, 1>::ConstRef indices,	
const typename optox::DTensor<T, 1>::ConstRef top_diff_real, 
const typename optox::DTensor<T, 1>::ConstRef top_diff_imag,
				typename optox::DTensor<T, 1>::Ref bottom_diff_real,
				typename optox::DTensor<T, 1>::Ref bottom_diff_imag,
              int time_in, int height_in, int width_in, int depth_in,
              int time_out, int height_out, int width_out, int depth_out,
              int kernel_t, int kernel_h, int kernel_w, int kernel_d,
              int stride_t, int stride_h, int stride_w, int stride_d,
              int channels, float alpha, float beta,
              int pad_t, int pad_h, int pad_w, int pad_d, int batch = 1)
{
	for (int n = 0; n < batch *time_in *height_in *width_in * channels; n++)

	{
		bottom_diff_real(n) = 0;
		bottom_diff_imag(n) = 0;

	}

int index = threadIdx.x + blockIdx.x *blockDim.x;

if (index < batch *time_out *height_out *width_out *depth_out *channels)
{
	int n = index;
	int ch = n % channels;
	n /= channels;
	int k = n % depth_out;
	n /= depth_out;
	int j = n % width_out;
	n /= width_out;
	int i = n % height_out;
	n /= height_out;
	int ti = n % time_out;
	n /= time_out;
	int b = n % batch;
	int b_coe = time_out *height_out *width_out *depth_out * channels;
	int t_coe = height_out *width_out *depth_out * channels;
	int i_coe = width_out *depth_out * channels;
	int j_coe = depth_out * channels;

	int b_in_coe = time_in *height_in *width_in *depth_in * channels;
	int t_in_coe = height_in *width_in *depth_in * channels;
	int h_in_coe = width_in *depth_in * channels;
	int w_in_coe = depth_in * channels;


	int out_idx = b *b_coe + ti *t_coe + i *i_coe + j *j_coe + k *channels + ch;
		

	int idx = indices(out_idx);
	n = idx;
	int ch_in = n % channels;
	n /= channels;
	int d = idx % depth_in;
	n /= depth_in;
	int w = n % width_in;
	n /= width_in;
	int h = n % height_in;
	n /= height_in;
	int t = n % time_in;
	n /= time_in;
	int b_i = n % batch;
	int bottom_idx = b_i *b_in_coe + t *t_in_coe + h *h_in_coe + w *w_in_coe + d *channels + ch_in;

	bottom_diff_real(bottom_idx) = top_diff_real(out_idx);
	bottom_diff_imag(bottom_idx) = top_diff_imag(out_idx);	
	}}
	



template < typename T>
	void optox::MaxPooling4d_Operator<T>::computeAdjoint(
		optox::OperatorOutputVector && outputs, const optox::OperatorInputVector &inputs

)
	{
		if (!with_indices_)
		{
			if (!batch_)
			{
				auto in_real = this->template getInput<T, 1> (0, inputs);
				auto in_imag = this->template getInput<T, 1> (1, inputs);

				auto out_real = this->template getInput<T, 1> (2, inputs);
				auto out_imag = this->template getInput<T, 1> (3, inputs);

				auto top_diff_real = this->template getInput<T, 1> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 1> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 1> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 1> (1, outputs);

			dim3 dim_block = dim3(16 * 16 * 4); 
			dim3 dim_grid = dim3(divUp(time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));

				maxpooling4d_complex_grad_backward<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				auto in_real = this->template getInput<T, 1> (0, inputs);
				auto in_imag = this->template getInput<T, 1> (1, inputs);

				auto out_real = this->template getInput<T, 1> (2, inputs);
				auto out_imag = this->template getInput<T, 1> (3, inputs);

				auto top_diff_real = this->template getInput<T, 1> (4, inputs);
				auto top_diff_imag = this->template getInput<T, 1> (5, inputs);

				auto bottom_diff_real = this->template getOutput<T, 1> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 1> (1, outputs);

				dim3 dim_block = dim3(16 * 16 * 4); 
				dim3 dim_grid = dim3(divUp(batch_*time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));
				
				maxpooling4d_complex_grad_backward_with_batch<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*in_real, *in_imag,
					*out_real, *out_imag, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
		else
		{
			if (!batch_)
			{
				auto indices = this->template getInput<T, 1> (0, inputs);
				auto top_diff_real = this->template getInput<T, 1> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 1> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 1> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 1> (1, outputs);

				dim3 dim_block = dim3(16 * 4 * 4); 
				dim3 dim_grid = dim3(divUp(batch_*time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));

				maxpooling4d_complex_grad_backward_with_indices <T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag, *bottom_diff_real, *bottom_diff_imag,
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_);

				OPTOX_CUDA_CHECK;
			}
			else
			{
				//batch_=True
				auto indices = this->template getInput<T, 1> (0, inputs);
				auto top_diff_real = this->template getInput<T, 1> (1, inputs);
				auto top_diff_imag = this->template getInput<T, 1> (2, inputs);

				auto bottom_diff_real = this->template getOutput<T, 1> (0, outputs);
				auto bottom_diff_imag = this->template getOutput<T, 1> (1, outputs);

				dim3 dim_block = dim3(4 * 8 * 8);
				dim3 dim_grid = dim3(divUp(batch_*time_out_*height_out_*width_out_*depth_out_*channels_, dim_block.x));

				maxpooling4d_complex_grad_backward_with_batch_and_indices<T> <<<dim_grid, dim_block, 0, this->stream_ >>> (*indices, *top_diff_real, *top_diff_imag,  *bottom_diff_real, *bottom_diff_imag,
				time_in_, height_in_, width_in_, depth_in_, 
				time_out_, height_out_, width_out_, depth_out_,
				kernel_t_, kernel_h_, kernel_w_, kernel_d_,
				stride_t_, stride_h_, stride_w_, stride_d_,
				channels_, alpha_, beta_, 
				pad_t_, pad_h_, pad_w_, pad_d_, batch_);

				OPTOX_CUDA_CHECK;
			}
		}
	}




#define REGISTER_OP(T)\
template class optox::MaxPooling4d_Operator<T> ;
OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
