#ifndef DFT_H_
#define DFT_H_

namespace dft{
	
	template <typename T>
	struct Complex {
		T re, im;
		Complex() {}
		Complex(T _re, T _im) :re(_re), im(_im) {}
	};

	int getOptimalDFTSize(int n);
	void copyMakeBorder(float* &src, int height, int width);

	int dft(const float* src, void* dst, int height, int width, int inv = 0);
	int idft(const float* src, void* dst, int height, int width, int inv = 0);
}


#endif // DFT_H_
