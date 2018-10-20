#ifndef DFT_H_
#define DFT_H_

#include <iostream>
#include <cassert>

namespace dft{
	
	template <typename T>
	struct Complex {
		T re, im;
		Complex() {}
		Complex(T _re, T _im) :re(_re), im(_im) {}
	};

	int getOptimalDFTSize(int n);
	void copyMakeBorder(const float* src, float* dst, int height, int width, int M, int N);

	int dft(const float* src, float* dst, int height, int width, int inv = 0);
	int idft(const float* src, float* dst, int height, int width);
}


#endif // DFT_H_
