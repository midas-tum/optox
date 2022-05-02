///@file py_averagepooling_operator.cpp
///@brief python wrappers for the average pooling operator
///@author Kaijie Mo<mokaijie5@gmail.com>
///@date 01.2021
#include <vector>
#include "py_utils.h"
#include "operators/averagepooling_operator.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template < typename T>
	std::tuple<py::array, py::array > forward1d(optox::AveragePooling1d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
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

			op.forward({ &output_real, &output_imag
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 2> (output_real), dTensorToNp<T, 2> (output_imag));
		}
		else if (np_input_real.ndim() == 3)
		{
			auto input_real = getDTensorNp<T, 3> (np_input_real);
			auto input_imag = getDTensorNp<T, 3> (np_input_imag);
			auto out_shape = input_real->size();
			
			out_shape[1]= height_out_;	// out height

			optox::DTensor<T, 3> output_real(out_shape);
			optox::DTensor<T, 3> output_imag(out_shape);


			op.forward({ &output_real, &output_imag
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (output_real), dTensorToNp<T, 3> (output_imag));
		}
	}

template < typename T>
	std::tuple<py::array, py::array > adjoint1d(optox::AveragePooling1d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{
		// parse the input tensors

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

template < typename T>
	std::tuple<py::array, py::array > forward2d(optox::AveragePooling2d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
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

			op.forward({ &output_real, &output_imag
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 3> (output_real), dTensorToNp<T, 3> (output_imag));
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

			op.forward({ &output_real, &output_imag
			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (output_real), dTensorToNp<T, 4> (output_imag));
		}
	}

template < typename T>
	std::tuple<py::array, py::array > adjoint2d(optox::AveragePooling2d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag)
	{
		// parse the input tensors

		int height_out_ = op.calculateHeightOut2D();
		int width_out_ = op.calculateWidthOut2D();

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

template < typename T>
	std::tuple<py::array, py::array > forward3d(optox::AveragePooling3d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
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

			op.forward({ &output_real, &output_imag},
			{input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 4> (output_real), dTensorToNp<T, 4> (output_imag));
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

			op.forward({ &output_real, &output_imag			},
			{
				input_real.get(), input_imag.get() });
			return std::make_tuple(dTensorToNp<T, 5> (output_real), dTensorToNp<T, 5> (output_imag));
		}
	}
template < typename T>
	std::tuple<py::array, py::array >  adjoint3d(optox::AveragePooling3d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag)
	{
		// parse the input tensors

		int height_out_ = op.calculateHeightOut3D();
		int width_out_ = op.calculateWidthOut3D();
		int depth_out_ = op.calculateDepthOut3D();

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
	
	
	template < typename T>
	std::tuple<py::array, py::array > forward4d(optox::AveragePooling4d_Operator<T> &op, py::array np_input_real, py::array np_input_imag)
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
 

		op.forward({ &output_real, &output_imag},
		{input_real.get(), input_imag.get() });
		return std::make_tuple(dTensorToNp<T, 1> (output_real), dTensorToNp<T, 1> (output_imag));


	}
template < typename T>
	std::tuple<py::array, py::array >  adjoint4d(optox::AveragePooling4d_Operator<T> &op, py::array np_input_real, py::array np_input_imag, py::array np_output_real, py::array np_output_imag, py::array np_top_diff_real, py::array np_top_diff_imag, py::array indices)
	{

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


template < typename T>
	void declare_op(py::module &m, const std::string &typestr)
	{
	    	std::string pyclass_name_1d = std::string("AveragePooling1d_") + typestr;
    		py::class_<optox::AveragePooling1d_Operator<T>>(m, pyclass_name_1d.c_str(), py::buffer_protocol(), py::dynamic_attr())			
    		.def(py::init<int&,   int&,   int&,    int&, float&, float&,     int&,   int&,    int&,int&,   std::string&>())
    .def("forward", forward1d<T>)
    .def("adjoint", adjoint1d<T>);


		std::string pyclass_name = std::string("AveragePooling2d_") + typestr;
		py::class_<optox::AveragePooling2d_Operator < T>> (m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int&, int&, int&, int&, int&, int&, int&, float&, float&, int&, int&, int&, int&, int&,int&,std::string&  >())
    .def("forward", forward2d<T>)
    .def("adjoint", adjoint2d<T>);

		std::string pyclass_name_3d = std::string("AveragePooling3d_") + typestr;
		py::class_<optox::AveragePooling3d_Operator < T>> (m, pyclass_name_3d.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int&, int&, int&, int&, int&, int&, int&, int&, int&, int&, float&, float&, int&, int&, int&, int&, int&, int&, int&,int&,  std::string&>())
    .def("forward", forward3d<T>)
    .def("adjoint", adjoint3d<T>);
    
    		std::string pyclass_name_4d = std::string("AveragePooling4d_") + typestr;
    		py::class_<optox::AveragePooling4d_Operator<T>>(m, pyclass_name_4d.c_str(), py::buffer_protocol(), py::dynamic_attr())
    			.def(py::init<int&,int&,int&,int&,   int&, int&, int&,int&,     int&, int&, int&,int&,     int&, float&, float&,    int&, int&,int&,int&,    int&, int&, int&,int&,     int&,int&,  std::string&>())
    .def("forward", forward4d<T>)
    .def("adjoint", adjoint4d<T>);
	}

PYBIND11_MODULE(py_averagepooling_operator, m)
{
	declare_op<float> (m, "float");
	declare_op<double> (m, "double");
}
