#include<mex.h>
#include<matrix.h>
#include"LSC.h"


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned char* R,* G,* B;
	int SuperpixelNum;
	int nRows;int nCols;
	double ratio;
	unsigned short* label;
	unsigned short* output;

	unsigned char* img;
	
	if(nrhs==2||nrhs==3)
	{
		if(mxGetNumberOfDimensions(prhs[0])!=3)
			mexErrMsgTxt("The input image must be in CIERGB form");
		if(mxGetClassID(prhs[0])!=mxUINT8_CLASS)
			mexErrMsgTxt("The input image must be in CIERGB form");
		nRows=mxGetM(prhs[0]);
		nCols=mxGetN(prhs[0])/3;

		img=(unsigned char*)mxGetPr(prhs[0]);
		SuperpixelNum=int(((double*)(mxGetPr(prhs[1])))[0]);

		if(nrhs==3)
			ratio=((double*)(mxGetPr(prhs[2])))[0];
		else
			ratio=0.1;

	}
	else if(nrhs>3)
		mexErrMsgTxt("Too many inputs!");
	else if(nrhs<2)
		mexErrMsgTxt("Too few inputs!");


	int pixel=nRows*nCols;
	R=new unsigned char[pixel];
	G=new unsigned char[pixel];
	B=new unsigned char[pixel];
	label=new unsigned short[pixel];

	for(int i=0;i<pixel;i++)
	{
		R[i]=img[i];
		G[i]=img[i+pixel];
		B[i]=img[i+pixel+pixel];
	}

	LSC(R,G,B,nCols,nRows,SuperpixelNum,ratio,label);

	plhs[0]=mxCreateNumericMatrix(nRows,nCols,mxUINT16_CLASS,mxREAL);
	output=(unsigned short*)mxGetPr(plhs[0]);
	for(int i=0;i<pixel;i++)
		output[i]=label[i];

	delete [] R;
	delete [] G;
	delete [] B;
	delete [] label;
}