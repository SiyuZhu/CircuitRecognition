/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
float maxval_f32() {return FLT_MAX;}
float minval_f32() {return -FLT_MAX;}
float nan_f32() { return NAN; }
float neg_inf_f32() { return -INFINITY; }
bool is_nan_f32(float x) {return x != x; }
float inf_f32() { return INFINITY; }
float float_from_bits(unsigned int x) {return as_float(x);}
char smod_char(char a, char b) {
char r = a % b;
if (r < 0) { r += b < 0 ? -b : b; }
return r;
}

short smod_short(short a, short b) {
short r = a % b;
if (r < 0) { r += b < 0 ? -b : b; }
return r;
}

int smod_int(int a, int b) {
int r = a % b;
if (r < 0) { r += b < 0 ? -b : b; }
return r;
}

long smod_long(long a, long b) {
long r = a % b;
if (r < 0) { r += b < 0 ? -b : b; }
return r;
}

char sdiv_char(char a, char b) {
char q = a / b;
char r = a - q*b;
char bs = b >> (8*sizeof(char) - 1);
char rs = r >> (8*sizeof(char) - 1);
return q - (rs&bs) + (rs&~bs);
}

short sdiv_short(short a, short b) {
short q = a / b;
short r = a - q*b;
short bs = b >> (8*sizeof(short) - 1);
short rs = r >> (8*sizeof(short) - 1);
return q - (rs&bs) + (rs&~bs);
}

int sdiv_int(int a, int b) {
int q = a / b;
int r = a - q*b;
int bs = b >> (8*sizeof(int) - 1);
int rs = r >> (8*sizeof(int) - 1);
return q - (rs&bs) + (rs&~bs);
}

long sdiv_long(long a, long b) {
long q = a / b;
long r = a - q*b;
long bs = b >> (8*sizeof(long) - 1);
long rs = r >> (8*sizeof(long) - 1);
return q - (rs&bs) + (rs&~bs);
}

#define sqrt_f32 sqrt 
#define sin_f32 sin 
#define cos_f32 cos 
#define exp_f32 exp 
#define log_f32 log 
#define abs_f32 fabs 
#define floor_f32 floor 
#define ceil_f32 ceil 
#define round_f32 round 
#define trunc_f32 trunc 
#define pow_f32 pow
#define asin_f32 asin 
#define acos_f32 acos 
#define tan_f32 tan 
#define atan_f32 atan 
#define atan2_f32 atan2
#define sinh_f32 sinh 
#define asinh_f32 asinh 
#define cosh_f32 cosh 
#define acosh_f32 acosh 
#define tanh_f32 tanh 
#define atanh_f32 atanh 
#define fast_inverse_f32 native_recip 
#define fast_inverse_sqrt_f32 native_rsqrt 
int halide_gpu_thread_barrier() {
  barrier(CLK_LOCAL_MEM_FENCE);
  return 0;
}
#define __address_space___shared __local

__kernel void _at_least_one_kernel(int x) { }
// Address spaces for kernel_convolution_s0_y___block_id_y
#define __address_space__convolution __global
#define __address_space__input __global
#define __address_space__kernel __global
__kernel void kernel_convolution_s0_y___block_id_y(
 const int _convolution_s0_c,
 const int _convolution_x_extent_realized,
 const int _convolution_x_min_realized,
 const int _convolution_y_extent_realized,
 const int _convolution_y_min_realized,
 const int _input_min_0,
 const int _input_min_1,
 const int _input_min_2,
 const int _input_min_2_required,
 const int _input_stride_1,
 const int _input_stride_2,
 const int _kernel_min_0,
 const int _kernel_min_1,
 const int _kernel_stride_1,
 const int _output_extent_0,
 const int _output_extent_1,
 const int _output_min_0,
 const int _output_min_1,
 const uchar _weight,
 __address_space__convolution uchar *_convolution,
 __address_space__input const uchar *_input,
 __address_space__kernel const uchar *_kernel,
 __address_space___shared int16* __shared)
{
 int _convolution_s0_y___block_id_y = get_group_id(1);
 int _0 = _convolution_s0_y___block_id_y * 8;
 int _1 = _0 + _output_min_1;
 int _2 = _output_min_1 + _output_extent_1;
 int _3 = _2 + -8;
 int _4 = min(_1, _3);
 int _convolution_s0_x___block_id_x = get_group_id(0);
 int ___thread_id_y = get_local_id(1);
 int ___thread_id_x = get_local_id(0);
 int _5 = _convolution_s0_x___block_id_x * 8;
 int _6 = _5 + _output_min_0;
 int _7 = _output_min_0 + _output_extent_0;
 int _8 = _7 + -8;
 int _9 = min(_6, _8);
 {
  float _sum[1];
  #define __address_space__sum __private
  // produce sum
  _sum[0] = float_from_bits(0 /* 0 */);
  // update sum
  for (int _sum_s1_r_y__r = 0; _sum_s1_r_y__r < 0 + 3; _sum_s1_r_y__r++)
  {
   for (int _sum_s1_r_x__r = 0; _sum_s1_r_x__r < 0 + 3; _sum_s1_r_x__r++)
   {
    float _10 = _sum[0];
    int _11 = 2 - _sum_s1_r_x__r;
    int _12 = 2 - _sum_s1_r_y__r;
    int _13 = _12 * _kernel_stride_1;
    int _14 = _11 + _13;
    int _15 = _kernel_min_1 * _kernel_stride_1;
    int _16 = _kernel_min_0 + _15;
    int _17 = _14 - _16;
    uchar _18 = _kernel[_17];
    float _19 = (float)(_18);
    int _20 = _9 + ___thread_id_x;
    int _21 = _20 - _sum_s1_r_x__r;
    int _22 = _4 + ___thread_id_y;
    int _23 = _22 - _sum_s1_r_y__r;
    int _24 = _23 + 2;
    int _25 = _24 * _input_stride_1;
    int _26 = _21 + _25;
    int _27 = _convolution_s0_c * _input_stride_2;
    int _28 = _26 + _27;
    int _29 = _input_min_1 * _input_stride_1;
    int _30 = _input_min_0 + _29;
    int _31 = _input_min_2 * _input_stride_2;
    int _32 = _30 + _31;
    int _33 = _28 - _32;
    int _34 = _33 + 2;
    uchar _35 = _input[_34];
    float _36 = (float)(_35);
    float _37 = _19 * _36;
    float _38 = _10 + _37;
    _sum[0] = _38;
   } // for _sum_s1_r_x__r
  } // for _sum_s1_r_y__r
  // consume sum
  float _39 = _sum[0];
  float _40 = (float)(_weight);
  float _41 = _39 / _40;
  int _42 = _9 + ___thread_id_x;
  int _43 = _42 - _convolution_x_min_realized;
  int _44 = _4 + ___thread_id_y;
  int _45 = _44 - _convolution_y_min_realized;
  int _46 = _45 * _convolution_x_extent_realized;
  int _47 = _43 + _46;
  int _48 = _convolution_s0_c - _input_min_2_required;
  int _49 = _convolution_x_extent_realized * _convolution_y_extent_realized;
  int _50 = _48 * _49;
  int _51 = _47 + _50;
  _convolution[_51] = (int)_41;
  #undef __address_space__sum
 } // alloc _sum
} // kernel kernel_convolution_s0_y___block_id_y
#undef __address_space__convolution
#undef __address_space__input
#undef __address_space__kernel
