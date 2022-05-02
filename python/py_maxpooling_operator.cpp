///@file py_maxpooling_operator.cpp
///@brief python wrappers for the maxpooling operator
///@author Kaijie Mo<mokaijie5@gmail.com>
///@date 01.2021
#include <vector>
#include "py_utils.h"
#include "operators/maxpooling_operator.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template < typename T>
	std::tuple<py::array, py::array, py::array > forward1d(optox::MaxPooling1d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
	{
		// parse the input tensors

		
		int height_out_ = op.calculateHeightOut1D();

		if (np_input_real.ndim() == 2)
		{				
			auto input_real = getDTensorNp<T, 2> (np_input_real);
			auto input_imag = getDTensorNp<T, 2> (np_input_imag);
			auto out_shape = input_real->size();		
			out_shape[0]= height_out_;	// out height
			optox::DTensor<T, 2> output_real(out_shape);
			optox::DTensor<T, 2> output_imag(out_shape);
			optox::DTensor<T, 2> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 2> (output_real), dTensorToNp<T, 2> (output_imag),  dTensorToNp<T, 2> (output_idx));
		}
		else if (np_input_real.ndim() == 3)
		{
			auto input_real = getDTensorNp<T, 3> (np_input_real);
			auto input_imag = getDTensorNp<T, 3> (np_input_imag);
			auto out_shape = input_real->size();
			
			out_shape[1]= height_out_;	// out height

			optox::DTensor<T, 3> output_real(out_shape);
			optox::DTensor<T, 3> output_imag(out_shape);
			optox::DTensor<T, 3> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (output_real), dTensorToNp<T, 3> (output_imag), dTensorToNp<T, 3> (output_idx) );
		}
	}

template < typename T>
	std::tuple<py::array, py::array > adjoint1d(optox::MaxPooling1d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{
		// parse the input tensors
		int with_indices=op.getIndices();
if (with_indices==0){

		if (np_input_real.ndim() == 2)
		{
			auto input_real = getDTensorNp<T, 2> (np_input_real);
			auto input_imag = getDTensorNp<T, 2> (np_input_imag);
			auto output_real = getDTensorNp<T, 2> (np_output_real);
			auto output_imag = getDTensorNp<T, 2> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 2> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 2> (np_top_diff_imag);
			auto bottom_shape = input_real->size();
			
		

			optox::DTensor<T, 2> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 2> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 2> (bottom_diff_real), dTensorToNp<T, 2> (bottom_diff_imag));
		}
		else if (np_input_real.ndim() == 3)
		{
			auto input_real = getDTensorNp<T, 3> (np_input_real);
			auto input_imag = getDTensorNp<T, 3> (np_input_imag);
			auto output_real = getDTensorNp<T, 3> (np_output_real);
			auto output_imag = getDTensorNp<T, 3> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 3> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 3> (np_top_diff_imag);
			auto bottom_shape = input_real->size();
		

			optox::DTensor<T, 3> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 3> bottom_diff_imag(bottom_shape);
			

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (bottom_diff_real), dTensorToNp<T, 3> (bottom_diff_imag));
		}
	}
	else{
	// indices!=NULL
	        int height_in_ = op.getHeightIn1D(); 	
			if (indices.ndim() == 2)
		{
			auto input_indices = getDTensorNp<T, 2> (indices);
			auto top_diff_real = getDTensorNp<T, 2> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 2> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			
			bottom_shape[0]=height_in_;
		

			optox::DTensor<T, 2> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 2> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 2> (bottom_diff_real), dTensorToNp<T, 2> (bottom_diff_imag));
		}
		else if (indices.ndim() == 3)
		{
			auto input_indices = getDTensorNp<T, 3> (indices);
			auto top_diff_real = getDTensorNp<T, 3> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 3> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			bottom_shape[1]=height_in_;
		
			optox::DTensor<T, 3> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 3> bottom_diff_imag(bottom_shape);
			

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (bottom_diff_real), dTensorToNp<T, 3> (bottom_diff_imag));
		}
	
	}
	
	}







template < typename T>
	std::tuple<py::array, py::array, py::array > forward2d(optox::MaxPooling2d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
	{
		// parse the input tensors
		int height_out_ = op.calculateHeightOut2D();
		int width_out_ = op.calculateWidthOut2D();

		if (np_input_real.ndim() == 3)
		{
			
			auto input_real = getDTensorNp<T, 3> (np_input_real);
			auto input_imag = getDTensorNp<T, 3> (np_input_imag);
			auto out_shape = input_real->size();		
			out_shape[0]= height_out_;	// out height
			out_shape[1]= width_out_;	// out weight
			optox::DTensor<T, 3> output_real(out_shape);
			optox::DTensor<T, 3> output_imag(out_shape);
			optox::DTensor<T, 3> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (output_real), dTensorToNp<T, 3> (output_imag),  dTensorToNp<T, 3> (output_idx));
		}
		else if (np_input_real.ndim() == 4)
		{
			auto input_real = getDTensorNp<T, 4> (np_input_real);
			auto input_imag = getDTensorNp<T, 4> (np_input_imag);
			auto out_shape = input_real->size();
			
			out_shape[1]= height_out_;	// out height
			out_shape[2]= width_out_;	// out weight


			optox::DTensor<T, 4> output_real(out_shape);
			optox::DTensor<T, 4> output_imag(out_shape);
			optox::DTensor<T, 4> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (output_real), dTensorToNp<T, 4> (output_imag), dTensorToNp<T, 4> (output_idx) );
		}
	}

template < typename T>
	std::tuple<py::array, py::array > adjoint2d(optox::MaxPooling2d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{
		// parse the input tensors
		int with_indices=op.getIndices();
if (with_indices==0){

		if (np_input_real.ndim() == 3)
		{
			auto input_real = getDTensorNp<T, 3> (np_input_real);
			auto input_imag = getDTensorNp<T, 3> (np_input_imag);
			auto output_real = getDTensorNp<T, 3> (np_output_real);
			auto output_imag = getDTensorNp<T, 3> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 3> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 3> (np_top_diff_imag);
			auto bottom_shape = input_real->size();
			
		

			optox::DTensor<T, 3> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 3> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (bottom_diff_real), dTensorToNp<T, 3> (bottom_diff_imag));
		}
		else if (np_input_real.ndim() == 4)
		{
			auto input_real = getDTensorNp<T, 4> (np_input_real);
			auto input_imag = getDTensorNp<T, 4> (np_input_imag);
			auto output_real = getDTensorNp<T, 4> (np_output_real);
			auto output_imag = getDTensorNp<T, 4> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 4> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 4> (np_top_diff_imag);
			auto bottom_shape = input_real->size();
		

			optox::DTensor<T, 4> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 4> bottom_diff_imag(bottom_shape);
			

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (bottom_diff_real), dTensorToNp<T, 4> (bottom_diff_imag));
		}
	}
	else{
	// indices!=NULL
	        int height_in_ = op.getHeightIn2D(); 
		int width_in_ = op.getWidthIn2D();
	
			if (indices.ndim() == 3)
		{
			auto input_indices = getDTensorNp<T, 3> (indices);
			auto top_diff_real = getDTensorNp<T, 3> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 3> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			
			bottom_shape[0]=height_in_;
			bottom_shape[1]=width_in_;
		

			optox::DTensor<T, 3> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 3> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (bottom_diff_real), dTensorToNp<T, 3> (bottom_diff_imag));
		}
		else if (indices.ndim() == 4)
		{
			auto input_indices = getDTensorNp<T, 4> (indices);
			auto top_diff_real = getDTensorNp<T, 4> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 4> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			bottom_shape[1]=height_in_;
			bottom_shape[2]=width_in_;
		

			optox::DTensor<T, 4> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 4> bottom_diff_imag(bottom_shape);
			

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (bottom_diff_real), dTensorToNp<T, 4> (bottom_diff_imag));
		}
	
	}
	
	}

template < typename T>
	std::tuple<py::array, py::array, py::array > forward3d(optox::MaxPooling3d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
	{
		// parse the input tensors


		int height_out_ = op.calculateHeightOut3D();
		int width_out_ = op.calculateWidthOut3D();
		int depth_out_ = op.calculateDepthOut3D();

		if (np_input_real.ndim() == 4)
		{
			auto input_real = getDTensorNp<T, 4> (np_input_real);
			auto input_imag = getDTensorNp<T, 4> (np_input_imag);
			auto out_shape = input_real->size();
			out_shape[0]= height_out_;	// out height
			out_shape[1]= width_out_;	// out weight
			out_shape[2]= depth_out_;	// out depth

			optox::DTensor<T, 4> output_real(out_shape);
			optox::DTensor<T, 4> output_imag(out_shape);
			optox::DTensor<T, 4> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (output_real), dTensorToNp<T, 4> (output_imag),dTensorToNp<T, 4> (output_idx) );
		}
		else if (np_input_real.ndim() == 5)
		{
			auto input_real = getDTensorNp<T, 5> (np_input_real);
			auto input_imag = getDTensorNp<T, 5> (np_input_imag);
			auto out_shape = input_real->size();
			out_shape[1]= height_out_;	// out height
			out_shape[2]= width_out_;	// out weight
			out_shape[3]= depth_out_;	// out depth

			optox::DTensor<T, 5> output_real(out_shape);
			optox::DTensor<T, 5> output_imag(out_shape);
			optox::DTensor<T, 5> output_idx(out_shape);

			op.forward({ &output_real, &output_imag, &output_idx
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 5> (output_real), dTensorToNp<T, 5> (output_imag), dTensorToNp<T, 5> (output_idx));
		}
	}
template < typename T>
	std::tuple<py::array, py::array >  adjoint3d(optox::MaxPooling3d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{int with_indices=op.getIndices();
	if (with_indices==0){
		// parse the input tensors

		if (np_input_real.ndim() == 4)
		{
			auto input_real = getDTensorNp<T, 4> (np_input_real);
			auto input_imag = getDTensorNp<T, 4> (np_input_imag);
			auto output_real = getDTensorNp<T, 4> (np_output_real);
			auto output_imag = getDTensorNp<T, 4> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 4> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 4> (np_top_diff_imag);
			auto bottom_shape = input_real->size();
			optox::DTensor<T, 4> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 4> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (bottom_diff_real), dTensorToNp<T, 4> (bottom_diff_imag));
		}
		else if (np_input_real.ndim() == 5)
		{
			auto input_real = getDTensorNp<T, 5> (np_input_real);
			auto input_imag = getDTensorNp<T, 5> (np_input_imag);
			auto output_real = getDTensorNp<T, 5> (np_output_real);
			auto output_imag = getDTensorNp<T, 5> (np_output_imag);
			auto top_diff_real = getDTensorNp<T, 5> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 5> (np_top_diff_imag);
			auto bottom_shape = input_real->size();

			optox::DTensor<T, 5> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 5> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 5> (bottom_diff_real), dTensorToNp<T, 5> (bottom_diff_imag));
		}
		}
		else{
	// indices!=NULL
		int height_in_ = op.getHeightIn3D();
		int width_in_ = op.getWidthIn3D();
		int depth_in_ = op.getDepthIn3D();
			if (indices.ndim() == 4)
		{
			auto input_indices = getDTensorNp<T, 4> (indices);
			auto top_diff_real = getDTensorNp<T, 4> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 4> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			bottom_shape[0]=height_in_;
			bottom_shape[1]=width_in_;
			bottom_shape[2]=depth_in_;
		

			optox::DTensor<T, 4> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 4> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (bottom_diff_real), dTensorToNp<T, 4> (bottom_diff_imag));
		}
		else if (indices.ndim() == 5)
		{
			auto input_indices = getDTensorNp<T, 5> (indices);
			auto top_diff_real = getDTensorNp<T, 5> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 5> (np_top_diff_imag);
			auto bottom_shape = input_indices->size();
			bottom_shape[1]=height_in_;
			bottom_shape[2]=width_in_;
			bottom_shape[3]=depth_in_;
		

			optox::DTensor<T, 5> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 5> bottom_diff_imag(bottom_shape);
			

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 5> (bottom_diff_real), dTensorToNp<T, 5> (bottom_diff_imag));
		}
	
	}
	}
	
template < typename T>
	std::tuple<py::array, py::array, py::array > forward4d(optox::MaxPooling4d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
	{
		// parse the input tensors


		int time_out_=op.calculateTimeOut4D();
		int height_out_=op.calculateHeightOut4D();
		int width_out_=op.calculateWidthOut4D();
		int depth_out_=op.calculateDepthOut4D();
		int channel_out_=op.getChannels4D(); 
		int batch_out_=op.getBatch4D();  

		auto input_real = getDTensorNp<T, 1> (np_input_real);
		auto input_imag = getDTensorNp<T, 1> (np_input_imag);
		
		auto out_shape = input_real->size();
		
		 if (batch_out_ !=0)
		{out_shape[0]= batch_out_ * time_out_ * height_out_* width_out_* depth_out_*channel_out_;}
		else{out_shape[0]= time_out_ * height_out_* width_out_* depth_out_*channel_out_;}	

		optox::DTensor<T, 1> output_real(out_shape);
		optox::DTensor<T, 1> output_imag(out_shape);
		optox::DTensor<T, 1> output_idx(out_shape);

		op.forward({ &output_real, &output_imag, &output_idx},
		{input_real.get(), input_imag.get() });
		return std::make_tuple(dTensorToNp<T, 1> (output_real), dTensorToNp<T, 1> (output_imag),dTensorToNp<T, 1> (output_idx) );


	}
template < typename T>
	std::tuple<py::array, py::array >  adjoint4d(optox::MaxPooling4d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{int with_indices=op.getIndices();
	if (with_indices==0){
		// parse the input tensors


		auto input_real = getDTensorNp<T, 1> (np_input_real);
		auto input_imag = getDTensorNp<T, 1> (np_input_imag);
		auto output_real = getDTensorNp<T, 1> (np_output_real);
		auto output_imag = getDTensorNp<T, 1> (np_output_imag);
		auto top_diff_real = getDTensorNp<T, 1> (np_top_diff_real);
		auto top_diff_imag = getDTensorNp<T, 1> (np_top_diff_imag);
		
		auto bottom_shape = input_real->size();
		optox::DTensor<T, 1> bottom_diff_real(bottom_shape);
		optox::DTensor<T, 1> bottom_diff_imag(bottom_shape);

		op.adjoint({ &bottom_diff_real, &bottom_diff_imag
		},
		{
			input_real.get(), input_imag.get(), output_real.get(), output_imag.get(), top_diff_real.get(), top_diff_imag.get() });
		return std::make_tuple(dTensorToNp<T, 1> (bottom_diff_real), dTensorToNp<T, 1> (bottom_diff_imag));
	

		}
		else{
	// indices!=NULL
			auto input_indices = getDTensorNp<T, 1> (indices);
			auto top_diff_real = getDTensorNp<T, 1> (np_top_diff_real);
			auto top_diff_imag = getDTensorNp<T, 1> (np_top_diff_imag);
			
			auto bottom_shape = input_indices->size();
			int b_in = op.getBatch4D();			
    			int t_in = op.getTimeIn4D();
    			int h_in = op.getHeightIn4D();
			int w_in = op.getWidthIn4D();
			int d_in = op.getDepthIn4D();
			int ch_in =op.getChannels4D();
			if (b_in!=0){
			    bottom_shape[0]=b_in * t_in * h_in * w_in * d_in *ch_in;
			    }
			    else{
			    bottom_shape[0]=t_in * h_in * w_in * d_in *ch_in;
			    }
		
			optox::DTensor<T, 1> bottom_diff_real(bottom_shape);
			optox::DTensor<T, 1> bottom_diff_imag(bottom_shape);

			op.adjoint({ &bottom_diff_real, &bottom_diff_imag
			},
			{
				input_indices.get(), top_diff_real.get(), top_diff_imag.get() });
			return std::make_tuple(dTensorToNp<T, 1> (bottom_diff_real), dTensorToNp<T, 1> (bottom_diff_imag));


	
	}
	}

template < typename T>
	void declare_op(py::module &m, const std::string &typestr)
	{
	    	std::string pyclass_name_1d = std::string("MaxPooling1d_") + typestr;
    		py::class_<optox::MaxPooling1d_Operator<T>>(m, pyclass_name_1d.c_str(), py::buffer_protocol(), py::dynamic_attr())			
    			.def(py::init<int&,   int&,   int&,    int&, float&, float&,     int&,   int&,    int&, int&,int&,  std::string&>())
    .def("forward", forward1d<T>)
    .def("adjoint", adjoint1d<T>);
	
		std::string pyclass_name_2d = std::string("MaxPooling2d_") + typestr;
		py::class_<optox::MaxPooling2d_Operator < T >> (m, pyclass_name_2d.c_str(), py::buffer_protocol(), py::dynamic_attr())
			.def(py::init<int&,int&,   int&,int&,   int&,int&,    int&, float&, float&,     int&,int&,   int&, int&,    int&, int&,int&,  std::string&>())
    .def("forward", forward2d<T>)
    .def("adjoint", adjoint2d<T>);

		std::string pyclass_name_3d = std::string("MaxPooling3d_") + typestr;
		py::class_<optox::MaxPooling3d_Operator < T >> (m, pyclass_name_3d.c_str(), py::buffer_protocol(), py::dynamic_attr())
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

PYBIND11_MODULE(py_maxpooling_operator, m)
{
	declare_op<float> (m, "float");
	declare_op<double> (m, "double");
}
