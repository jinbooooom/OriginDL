
#include <iostream>

#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

struct TestOp {
    template <typename T>
    HOST_DEVICE T operator()(T a, T b) const {
        return a + b;
    }
};

int main() {
    TestOp op;
    std::cout << "Result: " << op(3, 4) << std::endl;
    return 0;
}

