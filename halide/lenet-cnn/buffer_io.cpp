#include "buffer_io.hpp"

void init_buffer(buffer_t &buf, const int width, const int height, const int channels,
		 const int count, const int elem_size, uint8_t *const data,
		 bool host_dirty) {
  buf.extent[0] = width;
  buf.extent[1] = height;
  buf.extent[2] = channels;
  buf.extent[3] = count;

  buf.stride[0] = 1;
  buf.stride[1] = width;
  buf.stride[2] = width*height;
  buf.stride[3] = width*height*channels;

  buf.min[0] = 0;
  buf.min[1] = 0;
  buf.min[2] = 0;
  buf.min[3] = 0;

  buf.elem_size = elem_size;
  buf.host = data;
  buf.host_dirty = host_dirty;
}


#ifdef JPEGLIB_H
// NOTE: load_jpeg_buffer and save_jpeg_buffer were writting by referencing
// example.c in that came with libjpeg
void load_jpeg_buffer(std::string filename, buffer_t &buf) {
  FILE *f;
  JSAMPARRAY buffer;
  
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  if((f = fopen(filename.c_str(), "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    exit(1);
  }

  // allocate and initialize JPEG decompression object
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // specify data source
  jpeg_stdio_src(&cinfo, f);

  // read file parameters
  (void) jpeg_read_header(&cinfo, TRUE);

  // start decompressor
  (void) jpeg_start_decompress(&cinfo);
  
  // prepare output buffer
  buf.extent[3] = 1;
  buf.extent[2] = cinfo.output_components;
  buf.extent[1] = cinfo.output_height;
  buf.extent[0] = cinfo.output_width;
  buf.stride[2] = cinfo.output_height*cinfo.output_width;
  buf.stride[1] = cinfo.output_width;
  buf.stride[0] = 1;
  buf.elem_size = 1;
  buf.host_dirty = true;  
  buf.host = new uint8_t[cinfo.output_width*cinfo.output_height*
			cinfo.output_components];

  // store pixel values
  int row_stride = cinfo.output_width * cinfo.output_components;
  /* Make a one-row-high sample array that will go away when done with image */
  buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  while(cinfo.output_scanline < cinfo.output_height) {
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    for(uint8_t j = 0; j < cinfo.output_width; j++) {
      for(int i = 0; i < cinfo.output_components; i++) {
	*(buf.host + j + cinfo.output_scanline*cinfo.output_width +
	  i*cinfo.output_width*cinfo.output_height) = 
	  (buffer[0][j*cinfo.output_components + i]);
      }
    }
  }

  // finish decompressions
  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  
  return;
}
#endif

#ifdef PNG_H
void load_png_buffer(std::string filename, buffer_t &buf) {
	png_byte header[8];
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep *row_pointers;

	/* open file and test for it being a png */
	FILE *f = fopen(filename.c_str(), "rb");
	_assert(f, "File %s could not be opened for reading\n", filename.c_str());
	_assert(fread(header, 1, 8, f) == 8, "File ended before end of header\n");
	_assert(!png_sig_cmp(header, 0, 8), "File %s is not recognized as a PNG file\n", filename.c_str());

	/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	_assert(png_ptr, "png_create_read_struct failed\n");

	info_ptr = png_create_info_struct(png_ptr);
	_assert(info_ptr, "png_create_info_struct failed\n");

	_assert(!setjmp(png_jmpbuf(png_ptr)), "Error during init_io\n");

	png_init_io(png_ptr, f);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	int width = png_get_image_width(png_ptr, info_ptr);
	int height = png_get_image_height(png_ptr, info_ptr);
	int channels = png_get_channels(png_ptr, info_ptr);
	int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	// Expand low-bpp images to have only 1 pixel per byte (As opposed to tight packing)
	if (bit_depth < 8) {
		png_set_packing(png_ptr);
	}

	if (channels != 1) {
		buf.extent[0] = width;
		buf.extent[1] = height;
		buf.extent[2] = channels;

		buf.stride[0] = 1;
		buf.stride[1] = width;
		buf.stride[2] = width*height;
	} else {
		buf.extent[0] = width;
		buf.extent[1] = height;

		buf.stride[0] = 1;
		buf.stride[1] = width;
	}

	if (bit_depth == 8) {
		buf.elem_size = 1;
	} else {
		buf.elem_size = 2;
	}

	png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);

	// read the file
	_assert(!setjmp(png_jmpbuf(png_ptr)), "Error during read_image\n");

	row_pointers = new png_bytep[height];
	for (int y = 0; y < height; y++) {
		row_pointers[y] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
	}

	png_read_image(png_ptr, row_pointers);

	fclose(f);

	_assert((bit_depth == 8) || (bit_depth == 16), "Can only handle 8-bit or 16-bit pngs\n");

	// convert the data to uint8_t

	int c_stride = (channels == 1) ? 0 : width*height;
	uint8_t *img = (uint8_t*)malloc(width*height*channels*sizeof(uint8_t));
	uint8_t *ptr = img;
	_assert(ptr, "Error alocation memory for buffer.");
	if (bit_depth == 8) {
		for (int y = 0; y < height; y++) {
			uint8_t *srcPtr = (uint8_t *)(row_pointers[y]);
			for (int x = 0; x < width;x++) {
				for (int c = 0; c < channels; c++) {
					convert(*srcPtr++, ptr[c*c_stride]);
				}
				ptr++;
			}
		}
	} else if (bit_depth == 16) {
		for (int y = 0; y < height; y++) {
			uint8_t *srcPtr = (uint8_t *)(row_pointers[y]);
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; c++) {
					uint16_t hi = (*srcPtr++) << 8;
					uint16_t lo = hi | (*srcPtr++);
					convert(lo, ptr[c*c_stride]);
				}
				ptr++;
			}
		}
	}

	// Fill buffer_t
	buf.host = img;

	// clean up
	for (int y = 0; y < height; y++) {
		delete[] row_pointers[y];
	}
	delete[] row_pointers;

	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

	buf.host_dirty = true;
}
#endif

#ifdef JPEGLIB_H
void save_jpeg_buffer(buffer_t &buf, std::string filename, int quality) {
  FILE *f; // target file

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  // allocate and initialize JPEG compression object
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // specify data destination
  if((f = fopen(filename.c_str(), "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, f);

  // set parameters
  cinfo.image_width = buf.extent[0];
  cinfo.image_height = buf.extent[1];
  if((buf.extent[2] != 1) || (buf.extent[2] != 3)) {
    fprintf(stderr, "the buffer does not have a valid number of channels\n");
    exit(1);
  }
  cinfo.input_components = buf.extent[2];
  cinfo.in_color_space = ((buf.extent[2] == 1) ? JCS_GRAYSCALE : JCS_RGB);
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  // start compression
  jpeg_start_compress(&cinfo, TRUE);
  JSAMPROW row_pointer[buf.extent[0]*buf.extent[2]]; // points to JSAMPLE row
  while(cinfo.next_scanline < cinfo.image_height) {
    // copy pixel pointers to row_pointer
    for(int j = 0; j < buf.extent[0]; j++) {
      for(int i = 0; i < buf.extent[2]; i++)
	row_pointer[j*buf.extent[2] + i] = 
	  (JSAMPLE *) (buf.host + cinfo.next_scanline*buf.stride[1]
		    + j + i*buf.stride[2]);
    }
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  // finish compression
  jpeg_finish_compress(&cinfo);
  fclose(f);
  jpeg_destroy_compress(&cinfo);

  return;
}
#endif

#ifdef PNG_H
void save_png_buffer(buffer_t &buf, std::string filename) {
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep *row_pointers;
	png_byte color_type;

	int width, height, channels;
	width = buf.extent[0];
	height = buf.extent[1];
	channels = buf.extent[2];
	//im.copy_to_host();

	_assert(channels > 0 && channels < 5,
		"Can't write PNG files that have other than 1, 2, 3, or 4 channels\n");

	png_byte color_types[4] = {PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA,
														 PNG_COLOR_TYPE_RGB,  PNG_COLOR_TYPE_RGB_ALPHA
														};
	color_type = color_types[channels - 1];

	// open file
	FILE *f = fopen(filename.c_str(), "wb");
	_assert(f, "[write_png_file] File %s could not be opened for writing\n", filename.c_str());

	// initialize stuff
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	_assert(png_ptr, "[write_png_file] png_create_write_struct failed\n");

	info_ptr = png_create_info_struct(png_ptr);
	_assert(info_ptr, "[write_png_file] png_create_info_struct failed\n");

	_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during init_io\n");

	png_init_io(png_ptr, f);

	unsigned int bit_depth = 16;
	if (sizeof(uint8_t) == 1) {
		bit_depth = 8;
	}

	// write header
	_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing header\n");

	png_set_IHDR(png_ptr, info_ptr, width, height,
							bit_depth, color_type, PNG_INTERLACE_NONE,
							PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	row_pointers = new png_bytep[height];

	// im.copyToHost(); // in case the image is on the gpu

	int c_stride = (channels == 1) ? 0 : width*height;
	uint8_t *srcPtr = buf.host;

	for (int y = 0; y < height; y++) {
		row_pointers[y] = new png_byte[png_get_rowbytes(png_ptr, info_ptr)];
		uint8_t *dstPtr = (uint8_t *)(row_pointers[y]);
		if (bit_depth == 16) {
			// convert to uint16_t
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; c++) {
					uint16_t out;
					convert(srcPtr[c*c_stride], out);
					*dstPtr++ = out >> 8;
					*dstPtr++ = out & 0xff;
				}
				srcPtr++;
			}
		} else if (bit_depth == 8) {
			// convert to uint8_t
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channels; c++) {
					uint8_t out;
					convert(srcPtr[c*c_stride], out);
					*dstPtr++ = out;
				}
				srcPtr++;
			}
		} else {
			_assert(bit_depth == 8 || bit_depth == 16, "We only support saving 8- and 16-bit images.");
		}
	}

	// write data
	_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing bytes");

	png_write_image(png_ptr, row_pointers);

	// finish write
	_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during end of write");

	png_write_end(png_ptr, NULL);

	// clean up
	for (int y = 0; y < height; y++) {
		delete[] row_pointers[y];
	}
	delete[] row_pointers;

	fclose(f);

	png_destroy_write_struct(&png_ptr, &info_ptr);
}
#endif

void load_buffer(std::string filename, buffer_t &buf) {
	if (ends_with_ignore_case(filename, ".png")) {
		load_png_buffer(filename, buf);
	} else if (ends_with_ignore_case(filename, ".jpeg") ||
		   ends_with_ignore_case(filename, ".jpg")) {
	  load_jpeg_buffer(filename, buf);	  
	} else {
		_assert(false, "[load] unsupported file extension (png supported)");
	}
}

void save_buffer(buffer_t &buf, std::string filename, int quality) {
	if (ends_with_ignore_case(filename, ".png")) {
		save_png_buffer(buf, filename);
	} else if (ends_with_ignore_case(filename, ".jpeg") ||
		   ends_with_ignore_case(filename, ".jpg")) {
	  save_jpeg_buffer(buf, filename, quality);
	} else {
		_assert(false, "[save] unsupported file extension (png supported)");
	}
}
