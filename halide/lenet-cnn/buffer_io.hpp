// Simple IO helper to handle buffer_t structures used in Halide
// Enables read and store of jpeg and png images to and from buffer_t structs
//
// __OPENCL__ must be defined to enable helper functions that transmit data to
// and from a device. The user must include the OpenCL library in the main code
// before enabling the helper functions

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

#ifndef STATIC_IMAGE_LOADER_H
#define STATIC_IMAGE_LOADER_H

#include <png.h>
#include <stdio.h>
extern "C" {
#include <jpeglib.h>
}
#include <string>
#include <algorithm>
#include <string.h>

#define _assert(condition, ...) if (!(condition)) {fprintf(stderr, __VA_ARGS__); exit(-1);}

inline void convert(uint8_t in, uint8_t &out) {out = in;}
inline void convert(uint8_t in, uint16_t &out) {out = in << 8;}
inline void convert(uint8_t in, uint32_t &out) {out = in << 24;}
inline void convert(uint8_t in, float &out) {out = in/255.0f;}
inline void convert(uint8_t in, double &out) {out = in/255.0f;}
inline void convert(uint16_t in, uint8_t &out) {out = in >> 8;}
inline void convert(uint16_t in, uint16_t &out) {out = in;}
inline void convert(uint16_t in, uint32_t &out) {out = in << 16;}
inline void convert(uint16_t in, float &out) {out = in/65535.0f;}
inline void convert(uint16_t in, double &out) {out = in/65535.0f;}
inline void convert(uint32_t in, uint8_t &out) {out = in >> 24;}
inline void convert(uint32_t in, uint16_t &out) {out = in >> 16;}
inline void convert(uint32_t in, uint32_t &out) {out = in;}
inline void convert(uint32_t in, float &out) {out = in/4294967295.0f;}
inline void convert(uint32_t in, double &out) {out = in/4294967295.0f;}
inline void convert(float in, uint8_t &out) {out = (uint8_t)(in*255.0f);}
inline void convert(float in, uint16_t &out) {out = (uint16_t)(in*65535.0f);}
inline void convert(float in, uint32_t &out) {out = (uint16_t)(in*4294967295.0f);}
inline void convert(float in, float &out) {out = in;}
inline void convert(float in, double &out) {out = in;}
inline void convert(double in, uint8_t &out) {out = (uint8_t)(in*255.0f);}
inline void convert(double in, uint16_t &out) {out = (uint16_t)(in*65535.0f);}
inline void convert(double in, uint32_t &out) {out = (uint16_t)(in*4294967295.0f);}
inline void convert(double in, float &out) {out = in;}
inline void convert(double in, double &out) {out = in;}

void init_buffer(buffer_t &buf, const int width, const int height, const int channels,
		 const int count, const int elem_size, uint8_t *const data,
		 bool host_dirty = true);
inline void init_buffer(buffer_t &buf, const int width, const int height, const int channels,
			const int count, const long unsigned int elem_size, uint8_t *const data,
			bool host_dirty = true) {
  init_buffer(buf, width, height, channels, count, (int) elem_size, data, host_dirty);
}
inline void init_buffer(buffer_t &dst_buf, const buffer_t src_buf) {
  init_buffer(dst_buf, src_buf.extent[0], src_buf.extent[1], src_buf.extent[2], src_buf.extent[3],
	      src_buf.elem_size, src_buf.host, src_buf.host_dirty);
}
inline void init_buffer(buffer_t buf, const int width, const int height, const int channels, 
			uint8_t *const data) {
  init_buffer(buf, width, height, channels, 0, 1, data);
}
// ideally, the following two overloaded functions are not used since that utilize malloc,
// exacerbating memory management.
inline void init_buffer(buffer_t &buf, const int width, const int height, const int channels) {
  init_buffer(buf, width, height, channels, 0, 1, (uint8_t*)malloc(width*height*channels*sizeof(uint8_t)));
}
inline void init_buffer(buffer_t &buf, const int width, const int height, const int channels, 
			const int elem_size){
  init_buffer(buf, width, height, channels, 0, elem_size, (uint8_t*)malloc(width*height*channels*sizeof(uint8_t)));
}

// masks the given buffer so that dimension dim contains only layer n
inline void dimension_mask(buffer_t &buf, int dim, int n) {
  buf.extent[dim] = 1;
  buf.host = buf.host + n*buf.stride[dim];
}

inline bool ends_with_ignore_case(std::string a, std::string b) {
  if(a.length() < b.length()) {return false;}
  std::transform(a.begin(), a.end(), a.begin(), ::tolower);
  std::transform(a.begin(), a.end(), b.begin(), ::tolower);
  return a.compare(a.length()-b.length(), b.length(), b) == 0;
}

// load_jpeg_buffer and save_jpeg_buffer were implemented as described in
// example.c, which came with libjpeg
void load_jpeg_buffer(std::string filename, buffer_t &buf);
void save_jpeg_buffer(buffer_t &buf, std::string filename, int quality = 100);

void load_png_buffer(std::string filename, buffer_t &buf);
void save_png_buffer(buffer_t &buf, std::string filename);

void load_buffer(std::string filename, buffer_t &buf);
void save_buffer(buffer_t &buf, std::string filename, int quality = 100);

// TODO Can the folloing OpenCL code be defined in a cpp file? If so, realize.
#ifdef __OPENCL__
// A host <-> dev copy should be done with the fewest possible number
// of contiguous copies to minimize driver overhead. If our buffer_t
// has strides larger than its extents (e.g. because it represents a
// sub-region of a larger buffer_t) we can't safely copy it back and
// forth using a single contiguous copy, because we'd clobber
// in-between values that another thread might be using.  In the best
// case we can do a single contiguous copy, but in the worst case we
// need to individually copy over every pixel.
//
// This problem is made extra difficult by the fact that the ordering
// of the dimensions in a buffer_t doesn't relate to memory layout at
// all, so the strides could be in any order.
//
// We solve it by representing a copy job we need to perform as a
// device_copy struct. It describes a 4D array of copies to
// perform. Initially it describes copying over a single pixel at a
// time. We then try to discover contiguous groups of copies that can
// be coalesced into a single larger copy.

// The struct that describes a host <-> dev copy to perform.
#define MAX_COPY_DIMS 4
struct device_copy {
	uint64_t src, dst;
	// The multidimensional array of contiguous copy tasks that need to be done.
	uint64_t extent[MAX_COPY_DIMS];
	// The strides (in bytes) that separate adjacent copy tasks in each dimension.
	uint64_t stride_bytes[MAX_COPY_DIMS];
	// How many contiguous bytes to copy per task
	uint64_t chunk_size;
};

device_copy make_host_to_device_copy(const buffer_t *buf, cl_mem devbuf) {
	// Make a copy job representing copying the first pixel only.
	device_copy c;
	c.src = (uint64_t)buf->host;
	c.dst = (uint64_t)devbuf;
	c.chunk_size = buf->elem_size;
	for (int i = 0; i < MAX_COPY_DIMS; i++) {
		c.extent[i] = 1;
		c.stride_bytes[i] = 0;
	}

	if (buf->elem_size == 0) {
		// This buffer apparently represents no memory. Return a zero'd copy
		// task.
		device_copy zero = {0};
		return zero;
	}

	// Now expand it to copy all the pixels (one at a time) by taking
	// the extents and strides from the buffer_t. Dimensions are added
	// to the copy by inserting it s.t. the stride is in ascending order.
	for (int i = 0; i < 4 && buf->extent[i]; i++) {
		int stride_bytes = buf->stride[i] * buf->elem_size;
		// Insert the dimension sorted into the buffer copy.
		int insert;
		for (insert = 0; insert < i; insert++) {
			// If the stride is 0, we put it at the end because it can't be
			// folded.
			if (stride_bytes < c.stride_bytes[insert] && stride_bytes != 0) {
				break;
			}
		}
		for (int j = i; j > insert; j--) {
			c.extent[j] = c.extent[j - 1];
			c.stride_bytes[j] = c.stride_bytes[j - 1];
		}
		// If the stride is 0, only copy it once.
		c.extent[insert] = stride_bytes != 0 ? buf->extent[i] : 1;
		c.stride_bytes[insert] = stride_bytes;
	};

	// Attempt to fold contiguous dimensions into the chunk size. Since the
	// dimensions are sorted by stride, and the strides must be greater than
	// or equal to the chunk size, this means we can just delete the innermost
	// dimension as long as its stride is equal to the chunk size.
	while(c.chunk_size == c.stride_bytes[0]) {
		// Fold the innermost dimension's extent into the chunk_size.
		c.chunk_size *= c.extent[0];

		// Erase the innermost dimension from the list of dimensions to
		// iterate over.
		for (int j = 1; j < MAX_COPY_DIMS; j++) {
			c.extent[j-1] = c.extent[j];
			c.stride_bytes[j-1] = c.stride_bytes[j];
		}
		c.extent[MAX_COPY_DIMS-1] = 1;
		c.stride_bytes[MAX_COPY_DIMS-1] = 0;
	}
	return c;
}

device_copy make_device_to_host_copy(const buffer_t *buf, cl_mem devbuf) {
	// Just make a host to dev copy and swap src and dst
	device_copy c = make_host_to_device_copy(buf, devbuf);
	uint64_t tmp = c.src;
	c.src = c.dst;
	c.dst = tmp;
	return c;
}

int opencl_copy_to_device(buffer_t* buf, cl_mem devbuf, cl_command_queue command_queue) {
	
	device_copy c = make_host_to_device_copy(buf, devbuf);

	for (int w = 0; w < c.extent[3]; w++) {
		for (int z = 0; z < c.extent[2]; z++) {
			#ifdef ENABLE_OPENCL_11
			// OpenCL 1.1 supports stride-aware memory transfers up to 3D, so we
			// can deal with the 2 innermost strides with OpenCL.
			uint64_t off = z * c.stride_bytes[2] + w * c.stride_bytes[3];
			size_t offset[3] = { off, 0, 0 };
			size_t region[3] = { c.chunk_size, c.extent[0], c.extent[1] };

			cl_int err = clEnqueueWriteBufferRect(command_queue, (cl_mem)c.dst, CL_FALSE,
					offset, offset, region,
					c.stride_bytes[0], c.stride_bytes[1],
					c.stride_bytes[0], c.stride_bytes[1],
					(void *)c.src,
					0, NULL, NULL);

			if (err != CL_SUCCESS) {
				return err;
			}
			#else
			for (int y = 0; y < c.extent[1]; y++) {
				for (int x = 0; x < c.extent[0]; x++) {
					uint64_t off = (x * c.stride_bytes[0] +
							y * c.stride_bytes[1] +
							z * c.stride_bytes[2] +
							w * c.stride_bytes[3]);
					void *src = (void *)(c.src + off);
					void *dst = (void *)(c.dst + off);
					uint64_t size = c.chunk_size;

					cl_int err = clEnqueueWriteBuffer(command_queue, (cl_mem)c.dst,
							CL_FALSE, off, size, src, 0, NULL, NULL);
					if (err != CL_SUCCESS) {
						return err;
					}
				}
			}
			#endif
		}
	}

	return 0;
}

int opencl_copy_to_host(buffer_t* buf, cl_mem devbuf, cl_command_queue command_queue) {

	device_copy c = make_device_to_host_copy(buf, devbuf);

	for (int w = 0; w < c.extent[3]; w++) {
		for (int z = 0; z < c.extent[2]; z++) {
			#ifdef ENABLE_OPENCL_11
			// OpenCL 1.1 supports stride-aware memory transfers up to 3D, so we
			// can deal with the 2 innermost strides with OpenCL.
			uint64_t off = z * c.stride_bytes[2] + w * c.stride_bytes[3];

			size_t offset[3] = { off, 0, 0 };
			size_t region[3] = { c.chunk_size, c.extent[0], c.extent[1] };

			cl_int err = clEnqueueReadBufferRect(command_queue, (cl_mem)c.src, CL_FALSE,
					offset, offset, region,
					c.stride_bytes[0], c.stride_bytes[1],
					c.stride_bytes[0], c.stride_bytes[1],
					(void *)c.dst,
					0, NULL, NULL);

			if (err != CL_SUCCESS) {
				return err;
			}
			#else
			for (int y = 0; y < c.extent[1]; y++) {
				for (int x = 0; x < c.extent[0]; x++) {
					uint64_t off = (x * c.stride_bytes[0] +
					y * c.stride_bytes[1] +
					z * c.stride_bytes[2] +
					w * c.stride_bytes[3]);
					void *src = (void *)(c.src + off);
					void *dst = (void *)(c.dst + off);
					uint64_t size = c.chunk_size;

					cl_int err = clEnqueueReadBuffer(command_queue, (cl_mem)c.src,
					CL_FALSE, off, size, dst, 0, NULL, NULL);
					if (err != CL_SUCCESS) {
						return err;
					}
				}
			}
			#endif
		}
	}

	return 0;
}
#endif //OpenCL
#endif
