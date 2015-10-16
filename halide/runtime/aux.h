#ifndef _AUX_FUNCTIONS_HEADER_
#define _AUX_FUNCTIONS_HEADER_

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

void cl_assert(cl_int error) {
	if (error != CL_SUCCESS) {
		fprintf(stderr,"An API call returned an error\nError code: %d\n",error);
		exit(1);
	}
}

size_t buf_size(buffer_t *buf) {
	size_t size = buf->elem_size;
	for (size_t i = 0; i < sizeof(buf->stride) / sizeof(buf->stride[0]); i++) {
		size_t total_dim_size = buf->elem_size * buf->extent[i] * buf->stride[i];
		if (total_dim_size > size) {
			size = total_dim_size;
		}
	}
	return size;
}

void init_buffer(buffer_t &buf, const int width, const int height, const int channels) {

	buf.host = (uint8_t*)malloc(width*height*channels*sizeof(uint8_t));
	if (buf.host == NULL)
		fprintf(stderr, "Error while allocating memory space.\n");

	buf.extent[0] = width;
	buf.extent[1] = height;
	buf.extent[2] = channels;

	buf.stride[0] = 1;
	buf.stride[1] = width;
	buf.stride[2] = width*height;

	buf.min[0] = 0;
	buf.min[1] = 0;
	buf.min[2] = 0;

	buf.elem_size = 1;

	buf.host_dirty = true;
}

void init_buffer(buffer_t &buf, const int width, const int height, const int channels, uint8_t *data) {

	buf.host = data;

	buf.extent[0] = width;
	buf.extent[1] = height;
	buf.extent[2] = channels;

	buf.stride[0] = 1;
	buf.stride[1] = width;
	buf.stride[2] = width*height;

	buf.min[0] = 0;
	buf.min[1] = 0;
	buf.min[2] = 0;

	buf.elem_size = 1;

	buf.host_dirty = true;
}

void init_buffer(buffer_t &buf, const int width, const int height, const int channels, FILE *file_data) {

	int aux;
	buf.host = (uint8_t*)malloc(width*height*channels*sizeof(uint8_t));
	if (buf.host == NULL || file_data == NULL)
		fprintf(stderr, "Error while allocating memory space.\n");

	//Fill buffer with data
	int ch;
	(channels == 0)?ch=1:ch=channels;
	for (int c = 0; c < ch; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				fscanf(file_data,"%d", &aux);
				buf.host[c*height*width + y*width + x] = (uint8_t)aux;
			}
		}
	}

	buf.extent[0] = width;
	buf.extent[1] = height;
	buf.extent[2] = channels;

	buf.stride[0] = 1;
	buf.stride[1] = width;
	buf.stride[2] = width*height;

	buf.min[0] = 0;
	buf.min[1] = 0;
	buf.min[2] = 0;

	buf.elem_size = 1;

	buf.host_dirty = true;
}

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
#endif
