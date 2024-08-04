#include "dlzero.h"

int main()
{
	dl::NdArray data = { 1.0 };
	auto x = dl::Variable(data);
	x.Print();

	return 0;
}


