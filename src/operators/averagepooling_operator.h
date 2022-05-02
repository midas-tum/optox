#include "ioperator.h"

namespace optox
{


template <typename T>
    class OPTOX_DLLAPI AveragePooling1d_Operator : public IOperator
    {
public:

        /** Constructor.
         */
        AveragePooling1d_Operator(
            int& height_in, 
            int& kernel_h, 
            int& stride_h, 
            int& channels, float& alpha, float& beta,
            int& pad_h, 
            int& dilation_rate_h, 
            int& batch, int& ceil_mode, const std::string &padding_mode) : IOperator(),
            height_in_( height_in ),        
            kernel_h_( kernel_h ),
            stride_h_( stride_h ),
            channels_( channels ),
            alpha_( alpha ),
            beta_( beta ),
            pad_h_( pad_h ),         
            dilation_rate_h_(dilation_rate_h),
            batch_(batch), 
            padding_mode_(padding_mode) ,
            ceil_mode_(ceil_mode)                   

        {
        height_out_= calculateHeightOut1D();
        }

        /** Destructor */
        virtual ~AveragePooling1d_Operator()
        {
        }


        AveragePooling1d_Operator( AveragePooling1d_Operator const & )    = delete;
        void operator = ( AveragePooling1d_Operator const & )        = delete;
        
        
         int calculateOutShape(int size_in, int kernel_, int dilation_rate_, int stride_, int pad_) const
        {
        int effective_filter_size = (kernel_ - 1) * dilation_rate_ + 1;
	
	if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{ int size_out=0;
	if (this->ceil_mode_==1)
	{size_out = ceil((size_in + 2 * pad_ - effective_filter_size + stride_) / stride_);}
	else
	{size_out = (size_in + 2 * pad_ - effective_filter_size + stride_) / stride_;}
		return size_out;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int size_out = ceil(size_in / stride_);
		return size_out;}
	else return 0;

        }
        
        
         int calculateHeightOut1D() const
        {
		return calculateOutShape(this->height_in_, this->kernel_h_, this->dilation_rate_h_,this->stride_h_,this->pad_h_);

        }   

        
        int getHeightIn1D() const
        { return this->height_in_;}

        
        int calculateHeightIn1D() const
        { int effective_filter_size = (this->kernel_h_ - 1) * this->dilation_rate_h_ + 1;
        if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{int height_in = (this->height_out_-1)* this->stride_h_ + effective_filter_size ;
		return height_in;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int height_in = (this->height_out_-1)* this->stride_h_ + 1 ;
		return height_in;}
	else return 0;}
	




        


protected:
        virtual void computeForward( OperatorOutputVector && outputs,
    const OperatorInputVector &inputs );

        virtual void computeAdjoint( OperatorOutputVector && outputs,
    const OperatorInputVector &inputs );
    
       

        virtual unsigned int getNumOutputsForward()
        {
            return (2);
        }


        virtual unsigned int getNumInputsForward()
        {
            return(2);
        }


        virtual unsigned int getNumOutputsAdjoint()
        {
            return(2);
        }


        virtual unsigned int getNumInputsAdjoint()
        { return 6;}
           
        


private:

        int        height_in_;
        int        height_out_;
        int        kernel_h_;
        int        stride_h_;
        int        channels_;
        float      alpha_;
        float      beta_;
        int        pad_h_;       
        int        dilation_rate_h_;
        int        batch_;
        const      std::string padding_mode_;
        int        ceil_mode_;

        
        
    };





template <typename T>
class OPTOX_DLLAPI AveragePooling2d_Operator : public IOperator
{
public:

    /** Constructor.
     */
    AveragePooling2d_Operator(
        int& height_in, int& width_in,
        int& kernel_h, int& kernel_w,
        int& stride_h, int& stride_w,
        int& channels, float& alpha, float& beta,
        int& pad_h, int& pad_w,
        int& dilation_rate_h, int& dilation_rate_w,
        int& batch, int& ceil_mode,   const std::string& padding_mode) : IOperator(),
        height_in_( height_in ),
        width_in_( width_in ),
        kernel_h_( kernel_h ),
        kernel_w_( kernel_w ),
        stride_h_( stride_h ),
        stride_w_( stride_w ),
        channels_( channels ),
        alpha_( alpha ),
        beta_( beta ),
        pad_h_( pad_h ),
        pad_w_( pad_w ),
        dilation_rate_h_(dilation_rate_h),
        dilation_rate_w_(dilation_rate_w),
        batch_(batch),
        padding_mode_(padding_mode),
        ceil_mode_(ceil_mode) 
    {
        height_out_= calculateHeightOut2D();
        width_out_= calculateWidthOut2D();
    }

    /** Destructor */
    virtual ~AveragePooling2d_Operator()
    {
    }


    AveragePooling2d_Operator( AveragePooling2d_Operator const & )    = delete;
    void operator = ( AveragePooling2d_Operator const & )        = delete;
    
    
    
         int getHeightIn2D() const
        { return this->height_in_;}
        int getWidthIn2D() const
        { return this->width_in_;}
        
         int calculateOutShape(int size_in, int kernel_, int dilation_rate_, int stride_, int pad_) const
        {
        int effective_filter_size = (kernel_ - 1) * dilation_rate_ + 1;
	
	if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{ int size_out=0;
	if (this->ceil_mode_==1)
	{size_out = ceil((size_in + 2 * pad_ - effective_filter_size + stride_) / stride_);}
	else
	{size_out = (size_in + 2 * pad_ - effective_filter_size + stride_) / stride_;}
		return size_out;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int size_out = ceil(size_in / stride_);
		return size_out;}
	else return 0;

        }
        
        int calculateHeightOut2D() const
        {return calculateOutShape(this->height_in_, this->kernel_h_, this->dilation_rate_h_,this->stride_h_,this->pad_h_);}
        
                
        int calculateWidthOut2D() const
 	{return calculateOutShape(this->width_in_, this->kernel_w_, this->dilation_rate_w_,this->stride_w_,this->pad_w_);}
        


protected:
    virtual void computeForward( OperatorOutputVector && outputs,
                                 const OperatorInputVector &inputs );

    virtual void computeAdjoint( OperatorOutputVector && outputs,
                                 const OperatorInputVector &inputs );

    virtual unsigned int getNumOutputsForward()
    {
        return (2);
    }

    virtual unsigned int getNumInputsForward()
    {
        return(2);
    }

    virtual unsigned int getNumOutputsAdjoint()
    {
        return(2);
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return(6);
    }

private:

    int        height_in_;
    int        width_in_;
    int        height_out_;
    int        width_out_;
    int        kernel_h_;
    int        kernel_w_;
    int        stride_h_;
    int        stride_w_;
    int        channels_;
    float        alpha_;
    float        beta_;
    int        pad_h_;
    int        pad_w_;
    int        dilation_rate_h_;
    int        dilation_rate_w_;
    int        batch_;
    const std::string padding_mode_;
     int        ceil_mode_;
};


template <typename T>
class OPTOX_DLLAPI AveragePooling3d_Operator : public IOperator
{
public:


    /** Constructor.
     */
    AveragePooling3d_Operator( int& height_in, int& width_in, int& depth_in,
                               int& kernel_h, int& kernel_w, int& kernel_d,
                               int& stride_h, int& stride_w, int& stride_d,
                               int& channels, float& alpha, float& beta,
                               int& pad_h, int& pad_w, int& pad_d,
                               int& dilation_rate_h, int& dilation_rate_w, int& dilation_rate_d,
                               int& batch, int& ceil_mode, const std::string& padding_mode) : IOperator(),
        height_in_( height_in ),
        width_in_( width_in ),
        depth_in_( depth_in ),
        kernel_h_( kernel_h ),
        kernel_w_( kernel_w ),
        kernel_d_( kernel_d ),
        stride_h_( stride_h ),
        stride_w_( stride_w ),
        stride_d_( stride_d ),
        channels_( channels ),
        alpha_( alpha ),
        beta_( beta ),
        pad_h_( pad_h ),
        pad_w_( pad_w ),
        pad_d_( pad_d ),
        dilation_rate_h_(dilation_rate_h),
        dilation_rate_w_(dilation_rate_w),
        dilation_rate_d_(dilation_rate_d),
        batch_( batch ),
        padding_mode_(padding_mode),
        ceil_mode_(ceil_mode)
    {
        height_out_= calculateHeightOut3D();
        width_out_= calculateWidthOut3D();
        depth_out_ = calculateDepthOut3D();

    }


    /** Destructor */
    virtual ~AveragePooling3d_Operator()
    {
    }


    AveragePooling3d_Operator( AveragePooling3d_Operator const & )    = delete;
    void operator=( AveragePooling3d_Operator const & )        = delete;
    
             int getHeightIn3D() const
        { return this->height_in_;}
        int getWidthIn3D() const
        { return this->width_in_;}
         int getDepthIn3D() const
        { return this->depth_in_;}

                
         int calculateOutShape(int size_in, int kernel_, int dilation_rate_, int stride_, int pad_) const
        {
        int effective_filter_size = (kernel_ - 1) * dilation_rate_ + 1;
	
	if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{ int size_out=0;
	if (this->ceil_mode_==1)
	{size_out = ceil((size_in + 2 * pad_ - effective_filter_size + stride_) / stride_);}
	else
	{size_out = (size_in + 2 * pad_ - effective_filter_size + stride_) / stride_;}
		return size_out;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int size_out = ceil(size_in / stride_);
		return size_out;}
	else return 0;

        }

        int calculateHeightOut3D() const
        {return calculateOutShape(this->height_in_, this->kernel_h_, this->dilation_rate_h_,this->stride_h_,this->pad_h_);}
        
                
        int calculateWidthOut3D() const
 	{return calculateOutShape(this->width_in_, this->kernel_w_, this->dilation_rate_w_,this->stride_w_,this->pad_w_);}
 	
 	int calculateDepthOut3D() const
 	{return calculateOutShape(this->depth_in_, this->kernel_d_, this->dilation_rate_d_,this->stride_w_,this->pad_d_);}




protected:
    virtual void computeForward(     OperatorOutputVector && outputs,
                                     const OperatorInputVector &inputs);

    virtual void computeAdjoint(
        OperatorOutputVector && outputs,
        const OperatorInputVector &inputs);

    virtual unsigned int getNumOutputsForward()
    {
        return(2);
    }


    virtual unsigned int getNumInputsForward()
    {
        return(2);
    }

    virtual unsigned int getNumOutputsAdjoint()
    {
        return(2);
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return(6);
    }

private:
        int        height_in_;
        int        width_in_;
        int        depth_in_;
        int        height_out_;
        int        width_out_;
        int        depth_out_;
        int        kernel_h_;
        int        kernel_w_;
        int        kernel_d_;
        int        stride_h_;
        int        stride_w_;
        int        stride_d_;
        int        channels_;
        float        alpha_;
        float        beta_;
        int        pad_h_;
        int        pad_w_;
        int        pad_d_;
        int        dilation_rate_h_;
        int        dilation_rate_w_;
        int        dilation_rate_d_; 
        int        batch_;
        const std::string padding_mode_;
         int        ceil_mode_;
};


    template <typename T>
    class OPTOX_DLLAPI AveragePooling4d_Operator : public IOperator
    {
public:


        /** Constructor.
         */
        AveragePooling4d_Operator( int& time_in, int& height_in, int& width_in, int& depth_in,
                       int& kernel_t, int& kernel_h, int& kernel_w, int& kernel_d,
                      int& stride_t, int& stride_h, int& stride_w, int& stride_d,
                      int& channels, float& alpha, float& beta,
                      int& pad_t, int& pad_h, int& pad_w, int& pad_d, 
                      int& dilation_rate_t, int& dilation_rate_h, int& dilation_rate_w, int& dilation_rate_d,
                      int& batch,  int& ceil_mode,  const std::string &padding_mode ) : IOperator(),
                      
            time_in_(time_in),
            height_in_( height_in ),
            pad_d_( pad_d ),            
            width_in_( width_in ),
            depth_in_( depth_in ),
            kernel_t_( kernel_t ),
            kernel_h_( kernel_h ),
            kernel_w_( kernel_w ),
            kernel_d_( kernel_d ),
            stride_t_( stride_t ),
            stride_h_( stride_h ),
            stride_w_( stride_w ),
            stride_d_( stride_d ),
            channels_( channels ),
            alpha_( alpha ),
            beta_( beta ),
            pad_t_( pad_t),
            pad_h_( pad_h ), 
            pad_w_(pad_w ),               
            dilation_rate_t_(dilation_rate_t),         
            dilation_rate_h_(dilation_rate_h),
            dilation_rate_w_(dilation_rate_w),
            dilation_rate_d_(dilation_rate_d),
            batch_( batch ), 
            padding_mode_(padding_mode),
            ceil_mode_(ceil_mode)          
           
        {
        time_out_= calculateTimeOut4D();
        height_out_= calculateHeightOut4D();
        width_out_= calculateWidthOut4D();
        depth_out_ = calculateDepthOut4D();


        }


        /** Destructor */
        virtual ~AveragePooling4d_Operator()
        {
        }


        AveragePooling4d_Operator( AveragePooling4d_Operator const & )    = delete;
        void operator = ( AveragePooling4d_Operator const & )        = delete;

         int calculateOutShape(int size_in, int kernel_, int dilation_rate_, int stride_, int pad_) const
        {
        int effective_filter_size = (kernel_ - 1) * dilation_rate_ + 1;
	
	if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{ int size_out=0;
	if (this->ceil_mode_==1)
	{size_out = ceil((size_in + 2 * pad_ - effective_filter_size + stride_) / stride_);}
	else
	{size_out = (size_in + 2 * pad_ - effective_filter_size + stride_) / stride_;}
		return size_out;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int size_out = ceil(size_in / stride_);
		return size_out;}
	else return 0;

        }
        int calculateTimeOut4D() const
        {return calculateOutShape(this->time_in_, this->kernel_t_, this->dilation_rate_t_,this->stride_t_,this->pad_t_);}
            
        int calculateHeightOut4D() const
        {return calculateOutShape(this->height_in_, this->kernel_h_, this->dilation_rate_h_,this->stride_h_,this->pad_h_);}
                     
        int calculateWidthOut4D() const
 	{return calculateOutShape(this->width_in_, this->kernel_w_, this->dilation_rate_w_,this->stride_w_,this->pad_w_);}
 	
 	int calculateDepthOut4D() const
 	{return calculateOutShape(this->depth_in_, this->kernel_d_, this->dilation_rate_d_,this->stride_w_,this->pad_d_);}
        
        
         int getTimeIn4D() const
        { return this->time_in_;}
        
         int getHeightIn4D() const
        { return this->height_in_;}
        int getWidthIn4D() const
        { return this->width_in_;}
         int getDepthIn4D() const
        { return this->depth_in_;}
        
         int getBatch4D() const
        {return this->batch_;}
          int getChannels4D() const
        {return this->channels_;}
        
        int calculateTimeIn4D() const
        { int effective_filter_size = (this->kernel_t_ - 1) * this->dilation_rate_t_ + 1;
        if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{int time_in = (this->time_out_-1)* this->stride_t_ + effective_filter_size ;
	return time_in;}
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int time_in = (this->time_out_-1)* this->stride_t_ + 1 ;
		return time_in;}
	else return 0;}
        
        int calculateHeightIn4D() const
        { int effective_filter_size = (this->kernel_h_ - 1) * this->dilation_rate_h_ + 1;
        if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{int height_in = (this->height_out_-1)* this->stride_h_ + effective_filter_size ;
		return height_in;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int height_in = (this->height_out_-1)* this->stride_h_ + 1 ;
		return height_in;}
	else return 0;}
	
        int calculateWidthIn4D() const
        { int effective_filter_size = (this->kernel_w_ - 1) * this->dilation_rate_w_ + 1;
        if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{int width_in = (this->width_out_-1)* this->stride_w_ + effective_filter_size ;
		return width_in;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int width_in = (this->width_out_-1)* this->stride_w_ + 1 ;
		return width_in;}
	else return 0;}
	
	
	int calculateDepthIn4D() const
        { int effective_filter_size = (this->kernel_d_ - 1) * this->dilation_rate_d_ + 1;
        if (this->padding_mode_=="VALID" ||this->padding_mode_=="valid"||this->padding_mode_=="Valid")
	{int depth_in = (this->depth_out_-1)* this->stride_d_ + effective_filter_size ;
		return depth_in;}
		
	else if (this->padding_mode_=="SAME" ||this->padding_mode_=="same"||this->padding_mode_=="Same")
	{int depth_in = (this->depth_out_-1)* this->stride_d_ + 1 ;
		return depth_in;}
	else return 0;}





protected:
        virtual void computeForward(     OperatorOutputVector && outputs,
    const OperatorInputVector &inputs);

        virtual void computeAdjoint(     
    OperatorOutputVector && outputs,
    const OperatorInputVector &inputs);


        virtual unsigned int getNumOutputsForward()
        {
            return(2);
        }


        virtual unsigned int getNumInputsForward()
        {
            return(2);
        }


        virtual unsigned int getNumOutputsAdjoint()
        {
            return(2);
        }


        virtual unsigned int getNumInputsAdjoint()
        { return 6;}
        


private:

	int        time_in_;
        int        height_in_;       
        int        width_in_;
        int        depth_in_;
        
        int        time_out_;
        int        height_out_;
        int        width_out_;
        int        depth_out_;
        
        int        kernel_t_;
        int        kernel_h_;
        int        kernel_w_;
        int        kernel_d_;
        
        int        stride_t_;
        int        stride_h_;
        int        stride_w_;
        int        stride_d_;
        
        int        channels_;
        float        alpha_;
        float        beta_;
        
        int        pad_t_; 
        int        pad_h_; 
        int       pad_w_;   
        int        pad_d_;
        
        int        dilation_rate_t_;
        int        dilation_rate_h_;
        int        dilation_rate_w_;
        int        dilation_rate_d_; 
        int        batch_;
        const std::string padding_mode_;
        int        ceil_mode_;

        
        
    };
} /* namespace optox */
