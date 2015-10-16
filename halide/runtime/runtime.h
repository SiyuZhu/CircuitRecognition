#ifndef __USER_RUNTIME_H__
#define __USER_RUNTIME_H__

#include <stdio.h>
#include <fstream>
#include <string.h>
#include <CL/cl.h>

#ifndef BUFFER_T_DEFINED                                                        
#define BUFFER_T_DEFINED                                                        
#include <stdint.h>                                                             
typedef struct buffer_t {                                                       
  uint64_t dev;                                                                 
  uint8_t* host;                                                                
  int32_t extent[4];                                                            
  int32_t stride[4];                                                            
  int32_t min[4];                                                               
  int32_t elem_size;                                                            
  bool host_dirty;                                                              
  bool dev_dirty;                                                               
} buffer_t;                                                                     
#endif

#ifndef BUFFER_STRUCT_DEFINED
#define BUFFER_STRUCT_DEFINED
typedef struct buffer_index {
	uint8_t *buf_host;
	uint64_t buf_dev;
	int index;
} buffer_index;
#define BUF_INDEX_SIZE 100
#endif

static size_t bufsize(buffer_t *buf) {                       
	size_t size = buf->elem_size;                                               
	for (size_t i = 0; i < sizeof(buf->stride) / sizeof(buf->stride[0]); i++) { 
		size_t total_dim_size = buf->elem_size * buf->extent[i] * buf->stride[i];
		if (total_dim_size > size) {                                            
			size = total_dim_size;                                              
		}                                                                       
	}
	return size;
}

extern "C" {

static cl_context context;
static cl_int error;
static cl_uint uerror;

static buffer_index buf_index[100];
static int BufIndex = 0;

static FILE *api_src_file;
static FILE *api_header_file;

static int max_num_args = 0;
static int num_args = 0;
static int num_args_with_buffer = 0;
static int arg_sizes_runtime[100] = {0};
static int8_t* arg_is_buffer_runtime = NULL;
static uint64_t* args_runtime = NULL;
static int shared_mem_bytes_runtime = 0;

static int8_t* arg_is_constant = NULL;
static int bX, bY, bZ, tX, tY, tZ;

static int for_loop_index = 1;
static int matrix_index = 0;
static char *last_entry_name = NULL;

int halide_opencl_device_free(void *user_context, buffer_t* buf);
int halide_opencl_initialize_kernels(void *user_context, void **state_ptr,
			const char* src, int size);
int halide_opencl_device_release(void *user_context);
int halide_opencl_device_malloc(void *user_context, buffer_t* buf);
int halide_opencl_copy_to_device(void *user_context, buffer_t* buf);
int halide_opencl_copy_to_host(void *user_context, buffer_t* buf);
int halide_opencl_run(void *user_context, void *state_ptr,
			const char* entry_name,
			int blocksX, int blocksY, int blocksZ,
			int threadsX, int threadsY, int threadsZ,
			int shared_mem_bytes,
			size_t arg_sizes[], void* args[], int8_t arg_is_buffer[],
			int num_attributes,
			float* vertex_buffer,
			int num_coords_dim0, int num_coords_dim1);

int halide_copy_to_host(void *user_context, struct buffer_t *buf);
int halide_copy_to_device(void *user_context, struct buffer_t *buf);
int halide_device_free(void *user_context, struct buffer_t *buf);

}//extern C

#endif
