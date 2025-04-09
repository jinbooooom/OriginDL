#include "NumCpp.hpp"
#include <iostream>

int main()
{
    constexpr nc::uint32 numRows = 2;
    constexpr nc::uint32 numCols = 3;

    auto randArray1 = nc::random::rand<double>({numRows, numCols});
    std::cout << randArray1;

    auto randArray2 = nc::random::rand<double>({numRows, numCols});
    std::cout << randArray2;

    auto randArray3 = randArray1 + randArray2;
    std::cout << randArray3;

    nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} };
    nc::NdArray<int> b = { {11, 12}, {13, 14}, {15, 16} };
    auto c = a + b;
    std::cout << c;

    return 0;
}
