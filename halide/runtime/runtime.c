#include "runtime.h"

//TODO: Enable buffers with different elem_size?
//TODO: Test with more examples

void create_kernel();

int halide_opencl_device_free(void *user_context, buffer_t* buf) {
	buf->dev = 0;
	return 0;
}

int find_index_host(uint8_t *buf) {
	for (int i = 0; i < BUF_INDEX_SIZE; i++) {
		if (buf_index[i].buf_host == buf)
			return i;
	}
	return BUF_INDEX_SIZE;
}

int find_index_dev(uint64_t buf) {
	for (int i = 0; i < BUF_INDEX_SIZE; i++) {
		if (buf_index[i].buf_dev == buf && buf_index[i].buf_dev != 0)
			return i;
	}
	return BUF_INDEX_SIZE;
}

int printf_buffer() {
	for (int i = 0; i < BUF_INDEX_SIZE; i++) {
		fprintf(api_src_file,"%ld\n", buf_index[i].buf_host);
	}
}

int halide_opencl_initialize_kernels(void *user_context, void **state_ptr,
			const char* src, int size) {
	//Initialize host file
	last_entry_name = (char *)malloc(200*sizeof(char));
	strcpy(last_entry_name, "");

	//Open output file
	api_src_file = fopen("host.c", "w");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't create host file");
		return 1;
	}
	
	//Open kernel source file
	FILE *kernel_source_file = NULL;
	kernel_source_file = fopen("kernel.cl", "w");
	if (kernel_source_file == NULL) {
		fprintf(stderr,"Error: Can't create auxiliar header file");
		return 1;
	}
	fprintf(kernel_source_file, "%s", src);
	fclose(kernel_source_file);

	cl_device_id device_id;
	cl_platform_id platform_id;
	clGetPlatformIDs(1, &platform_id, &uerror);
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &uerror);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);

	//Initialize header and source
	fprintf(api_src_file,
	"#include <stdio.h>\n"
	"#include <stdlib.h>\n"
	"#include <pthread.h>\n"
	"#include <CL/cl.h>\n"
	"#include \"./aux.h\"\n"
	"#define MAX_SOURCE_SIZE %d\n\n",size);

	fprintf(api_src_file,
	"int main() {\n"
	"\t//Setting up variables\n"
	"\tcl_device_id device_id = NULL;\n"
	"\tcl_context context = NULL;\n"
	"\tcl_command_queue cmd_queue = NULL;\n"
	"\tcl_program program = NULL;\n"
	"\tcl_kernel kernel = NULL;\n"
	"\tcl_platform_id platform_id = NULL;\n"
	"\tcl_uint ret_num_platforms, ret_num_devices;\n"
	"\tcl_int error;\n\n"
	"\t//Reading kernel source\n"
	"\tFILE *kernel_file = NULL;\n"
	"\tFILE *file_data = NULL;\n"
	"\tchar *source_str = NULL;\n"
	"\tsize_t source_size;\n"
	"\tsize_t alloc_size;\n"
	"\tkernel_file = fopen(\"./kernel.cl\",\"r\");\n"
	"\tif (kernel_file == NULL) {\n"
	"\t\tfprintf(stderr, \"Failed to load kernel source.\");\n"
	"\t\texit(1);\n"
	"\t}\n"
	"\tsource_str = (char*)malloc(MAX_SOURCE_SIZE*sizeof(char));\n"
	"\tsource_size = fread(source_str, 1, MAX_SOURCE_SIZE, kernel_file);\n"
	"\tfclose(kernel_file);\n\n"
	"\t//Launch kernel variables\n"
	"\tint blocksX, blocksY, blocksZ;\n"
	"\tint threadsX, threadsY, threadsZ;\n"
	"\tsize_t global_dim[3];\n"
	"\tsize_t local_dim[3];\n\n");

	fprintf(api_src_file,
	"\terror = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);\n"
	"\tcl_assert(error);\n"
	"\terror = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);\n"
	"\tcl_assert(error);\n"
	"\t//Creating Context\n"
	"\tcontext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);\n"
	"\tcl_assert(error);\n"
	"\tcmd_queue = clCreateCommandQueue(context, device_id, 0, &error);\n"
	"\tcl_assert(error);\n"
	"\tprogram = clCreateProgramWithSource(context, 1, (const char **)&source_str"
	", (const size_t *)&source_size, &error);\n"
	"\tcl_assert(error);\n"
	"\terror = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);\n"
	"\tcl_assert(error);\n\n");

	fclose(api_src_file);
	api_src_file == NULL;
	return CL_SUCCESS;
}

int halide_opencl_device_release(void *user_context) {
	//Finish application
	
	//Open output file
	api_src_file = fopen("host.c", "a");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't create host file");
		return 1;
	}

	fprintf(api_src_file,
	"\t//Finish application\n"
	"\tclReleaseProgram(program);\n"
	"\tclReleaseCommandQueue(cmd_queue);\n"
	"\tclReleaseContext(context);\n"
	"\treturn 0;\n"
	"}\n");

	fclose(api_src_file);
	api_src_file == NULL;
}

int halide_opencl_device_malloc(void *user_context, buffer_t* buf) {
	
	if (find_index_host(buf->host) != BUF_INDEX_SIZE)
		return 0;
	

	//Open output file
	api_src_file = fopen("host.c", "a");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't update host file");
		return 1;
	}

	size_t buffer_size = bufsize(buf);
	//Allocate buffer in the GPU for Halide control
	//it's necessary for some runtime functions in Halide
	//and for our control as well
	cl_mem dev_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &error);

	//Read data from buffer and print to a file
	FILE *data_stream;
	char *data;
	char name[30];
	sprintf(name, "Input_Buf%d.dat",BufIndex);

	//Print buffer data to a file so we can read later
	data_stream = fopen(name,"w");
	for (int c = 0; c < ((buf->extent[2] == 0)?1:buf->extent[2]); c++) {
		int stride_c = buf->stride[2];
		for (int y = 0; y < buf->extent[1]; y++) {
			int stride_y = buf->stride[1];
			for (int x = 0; x < buf->extent[0]; x++) {
				fprintf(data_stream,"%d ",buf->host[c*buf->extent[0]*buf->extent[1]
				+ y*buf->extent[0] + x]);
			}
			fprintf(data_stream,"\n");
		}
		fprintf(data_stream,"\n");
	}
	fclose(data_stream);

	fprintf(api_src_file,
	"\t//Allocate memory in device and fill buffer if needed\n"
	"\tfile_data = fopen(\"Input_Buf%d.dat\",\"r\");\n"
	"\tif (file_data == NULL) {\n"
	"\t\tprintf(\"Error: Data file doesn't exist.\\n\");\n"
	"\t\texit(1);\n"
	"\t}\n"
	"\tbuffer_t Input_Buf%d = {0};\n"
	"\tinit_buffer(Input_Buf%d, %d, %d, %d, file_data);\n"
	"\tfclose(file_data);\n"
	"\tcl_mem Buf%d;\n"
	"\talloc_size = buf_size(&Input_Buf%d);\n"
	"\tBuf%d = clCreateBuffer(context, CL_MEM_READ_WRITE, alloc_size, NULL, &error);\n"
	"\tcl_assert(error);\n"
	, BufIndex
	, BufIndex, BufIndex, buf->extent[0], buf->extent[1], buf->extent[2]
	, BufIndex, BufIndex, BufIndex);

	//Insert buffer in the buffer_index structure
	buf_index[BufIndex].buf_host = buf->host;
	buf_index[BufIndex].buf_dev = (uint64_t)dev_ptr;
	buf_index[BufIndex].index = BufIndex;
	//Change buf->dev to show that now the buffer is allocated in the device
	void *ptr = buf;
	printf("Buf%d: %p\n", BufIndex, (void*)ptr);
	printf("Dev: %p\n\n", (void*)dev_ptr);
	buf->dev = (uint64_t)dev_ptr;
	BufIndex++;

	fclose(api_src_file);
	api_src_file == NULL;

	return CL_SUCCESS;
}

int halide_opencl_copy_to_device(void *user_context, buffer_t* buf) {
	//Open output file
	api_src_file = fopen("host.c", "a");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't update host file");
		return 1;
	}

	fprintf(api_src_file,
	"\topencl_copy_to_device(&Input_Buf%d, Buf%d, cmd_queue);\n"
	, BufIndex-1, BufIndex-1);

	fclose(api_src_file);
	api_src_file == NULL;

	return CL_SUCCESS;
}

int halide_copy_to_device(void *user_context, struct buffer_t *buf) {
	//Check if buffer is allocated in the device, if not, do it
	//Check to see if it's necessary to copy the buffer to device
	if (buf->dev == 0) {
		halide_opencl_device_malloc(user_context, buf);
	}

	if (buf->host_dirty) {
		halide_opencl_copy_to_device(user_context, buf);
	}
	return 0;

}

int halide_copy_to_host(void *user_context, struct buffer_t *buf){
	//Only if the buffer was changed by the device we can copy
	if (buf->dev_dirty == true) {
		halide_opencl_copy_to_host(user_context, buf);
	}

}

int halide_device_free(void *user_context, struct buffer_t *buf) {
	//Release buffer from device
	buf->dev_dirty = false;
	return 0;
	
}

int halide_opencl_copy_to_host(void *user_context, buffer_t* buf) {
	//It was necessary to change the halide_copy_to_host function
	//because we don't use the halide_device_interface struct
	create_kernel();
	//Open output file
	api_src_file = fopen("host.c", "a");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't update host file");
		return 1;
	}

	int index = find_index_host(buf->host);
	//int index = buf->dev - 10;

	fprintf(api_src_file,
	"\t//Read result from GPU\n"
	"\topencl_copy_to_host(&Input_Buf%d, Buf%d, cmd_queue);\n\n"
	"\tfile_data = fopen(\"Input_Buf%d.dat\",\"w\");\n"
	"\tif (file_data == NULL) exit(1);\n"
	"\tfor (int c = 0; c < Input_Buf%d.extent[2]; c++) {\n"
	"\t\tfor (int y = 0; y < Input_Buf%d.extent[1]; y++) {\n"
	"\t\t\tfor (int x = 0; x < Input_Buf%d.extent[0]; x++) {\n"
	"\t\t\t\tlong int stride_buf = c*Input_Buf%d.stride[2] + y*Input_Buf%d.stride[1];\n"
	"\t\t\t\tfprintf(file_data,\"%%d \",Input_Buf%d.host[stride_buf + x]);\n"
	"\t\t\t}\n"
	"\t\t\tfprintf(file_data,\"\\n\");\n"
	"\t\t}\n"
	"\t\tfprintf(file_data,\"\\n\");\n"
	"\t}\n\n"
	,index, index, index, index, index, index, index, index, index);

	fclose(api_src_file);
	api_src_file == NULL;

	return 0;
}

int halide_opencl_run(void *user_context, void *state_ptr,
			const char* entry_name,
			int blocksX, int blocksY, int blocksZ,
			int threadsX, int threadsY, int threadsZ,
			int shared_mem_bytes,
			size_t arg_sizes[], void* args[], int8_t arg_is_buffer[],
			int num_attributes,
			float* vertex_buffer,
			int num_coords_dim0, int num_coords_dim1) {
	
	if (!strcmp(entry_name,last_entry_name)) {
		for_loop_index++;
	} else {
		if (matrix_index != 0) {
			create_kernel();
		} else {
			matrix_index++;
		}

		//Start a new kernel loop
		printf("Entry: %s\n", entry_name);
		strcpy(last_entry_name, entry_name);
		for_loop_index = 1;
		
		//Get information about the kernel
		num_args = 0;
		num_args_with_buffer = 0;
		for (int i = 0; arg_sizes[i] != 0; i++) {
			if (arg_is_buffer[i] == false) {
				num_args++;
			}
			num_args_with_buffer++;
		}
		
		if (arg_is_buffer_runtime != NULL) {
			free(arg_is_buffer_runtime);
			arg_is_buffer_runtime = NULL;
		}
		arg_is_buffer_runtime = (int8_t*)malloc(sizeof(int8_t)*num_args_with_buffer);
		for (int i = 0; i < num_args_with_buffer; i++) {
			arg_is_buffer_runtime[i] = arg_is_buffer[i];
		}

		if (args_runtime != NULL) {
			free(args_runtime);
			args_runtime = NULL;
		}
		args_runtime = (uint64_t *)malloc(sizeof(uint64_t)*num_args_with_buffer);
	
		if (arg_is_constant != NULL) {
			free(arg_is_constant);
			arg_is_constant = NULL;
		}	
		arg_is_constant = (int8_t*)malloc(sizeof(int8_t)*num_args_with_buffer);

		for (int i = 0; i < num_args_with_buffer; i++) {
			arg_sizes_runtime[i] = arg_sizes[i];
			if (arg_sizes[i] == 4) {
				args_runtime[i] = *((uint64_t *)args[i]) & 0xFFFFFFFF;
				//printf("%d: %ld\n", i, *((uint64_t*)args[i]) & 0xFFFFFFFF);
			} else if (arg_sizes[i] == 1) {
				args_runtime[i] = *((uint64_t *)args[i]) & 0xFF;
				//printf("%d: %ld\n", i, *((uint64_t*)args[i]) & 0xFF);
			} else {
				args_runtime[i] = *((uint64_t *)args[i]) & 0xFFFFFFFFFFFFFFFF;
				//printf("%d: %ld\n", i, *((uint64_t*)args[i]) & 0xFFFFFFFFFFFFFFFF);
			}
		}
		shared_mem_bytes_runtime = shared_mem_bytes;
		bX = blocksX; bY = blocksY; bZ = blocksZ;
		tX = threadsX; tY = threadsY; tZ = threadsZ;

	}

	return CL_SUCCESS;
}

void create_kernel() {
	
	//Open output file
	api_src_file = fopen("host.c", "a");
	if (api_src_file == NULL) {
		fprintf(stderr,"Error: Can't update host file");
		exit(1);
	}
	
	
	fprintf(api_src_file,
	"\tglobal_dim[0] = %d*%d; global_dim[1] = %d*%d; global_dim[2] = %d*%d;\n"
	"\tlocal_dim[0] = %d; local_dim[1] = %d; local_dim[2] = %d;\n\n"
	"\tfor (int i = 0; i < %d; i++) {\n"
	"\t\t\n"
	, bX, tX, bY, tY, bZ, tZ
	,	tX, tY, tZ
	, for_loop_index);

	fprintf(api_src_file,
	"\n\tkernel = clCreateKernel(program, \"%s\", &error);\n"
	"\tcl_assert(error);\n"
	, last_entry_name);
	//Set arguments
	int i = 0;
	for(i = 0; i < num_args_with_buffer; i++) {
		
		if (arg_is_buffer_runtime[i]) {
			//Identify buffer and give appropriate index
			if (args_runtime[i] != NULL) {
				int aux = find_index_dev(args_runtime[i]);
				fprintf(api_src_file,
				"\t\terror = clSetKernelArg(kernel, %d, %d, (void*)&Buf%d);\n"
				, i, (int)arg_sizes_runtime[i], aux/*(uint64_t)args[i]*/);
			} else {
				fprintf(api_src_file,
				"\t\terror = clSetKernelArg(kernel, %d, %d, NULL);\n"
				, i, (int)arg_sizes_runtime[i]);
			}	
		} else {
			fprintf(api_src_file,
			"\t\terror = clSetKernelArg(kernel, %d, %d, (void*)&arg%d_%d);\n"
			, i, (int)arg_sizes_runtime[i], matrix_index, i);
		}
		fprintf(api_src_file,"\t\tcl_assert(error);\n");
	}
	
	//Shared mem buffer last
	fprintf(api_src_file,
	"\t\terror = clSetKernelArg(kernel, %d, %d, NULL);\n"
	"\t\tcl_assert(error);\n"
	, i, (shared_mem_bytes_runtime > 0) ? shared_mem_bytes_runtime : 1);

	//Run the kernel
	fprintf(api_src_file,
	"\t\t//Launch Kernel\n"
	"\t\terror = clEnqueueNDRangeKernel(cmd_queue, kernel, 3, NULL, global_dim,"
	" local_dim, 0, NULL, NULL);\n"
	"\t\tcl_assert(error);\n\n"
	"\t\t//Releasing Kernel\n"
	"\t\tclReleaseKernel(kernel);\n"
	"\t\terror = clFinish(cmd_queue);\n"
	"\t\tcl_assert(error);\n\n");

	//Close loop
	fprintf(api_src_file,
	"\t}\n\n"
	);

	matrix_index++;
	
	fclose(api_src_file);
	api_src_file == NULL;
	return;
}
