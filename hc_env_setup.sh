# Environment setup for Heterogenous Computing Project on zhang-01
# Source this or put it in your .bashrc
# Do not change the order of the exports

if [[ `hostname -s` = zhang* ]]; then
  echo "Setting up zhang server environment"
  
  # Caffe Python files
  export PYTHONPATH=/work/zhang/common/tools/caffe/distribute/python:$PYTHONPATH

  #------------------------------------------------------------------
  # Various Tools (including opencv)
  #------------------------------------------------------------------
  USR=/work/zhang/common/usr
  export CMAKE_ROOT=$USR/bin
  export PATH=$USR/bin:$PATH
  export LIBRARY_PATH=$USR/lib64:$USR/lib:/lib64:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$USR/lib64:$USR/lib:/lib64:$LD_LIBRARY_PATH
  export LD_RUN_PATH=$USR/lib64:$USR/lib:$LD_RUN_PATH
  export CPATH=$USR/include:$CPATH
  export PKG_CONFIG_PATH=$USR/lib/pkgconfig:$USR/share/pkgconfig:$PKG_CONFIG_PATH

  #------------------------------------------------------------------
  # Env for Halide
  #------------------------------------------------------------------
  TOOLS="/work/zhang/common/tools"
  HALIDE="halide-latest/Halide"
  HALIDE_BIN=$TOOLS/$HALIDE/bin
  HALIDE_INC=$TOOLS/$HALIDE/include
  export LIBRARY_PATH=$HALIDE_BIN:$LIBRARY_PATH
  export LD_LIBRARY_PATH=$HALIDE_BIN:$LD_LIBRARY_PATH
  export CPATH=$HALIDE_INC:$CPATH
  alias ghl="g++ -lHalide -lpng -lpthread -ldl"
  
  #------------------------------------------------------------------
  # Env for LLVM
  #------------------------------------------------------------------
  export PATH="/work/zhang/common/tools/llvm/3.4/bin:$PATH"

  # Example OpenCV compile:
  #
  # Compile source files
  # > g++ main.cpp -c `pkg-config --cflags opencv`
  #
  # Link objects files
  # > g++ main.o -o main `pkg-config --libs opecnv`
  # 
  # The pkg-config call must be at the end of the command

  # Example Halide compile
  #
  # > g++ main.cpp -o main -lHalide -lpng -lpthread -ldl


  #------------------------------------------------------------------
  # Env for Caffe
  #------------------------------------------------------------------
  export PYTHONPATH=/work/zhang/common/tools/caffe/distribute/python:$PYTHONPATH
  export CAFFE_ROOT=/work/zhang/common/tools/caffe
fi

if [[ `hostname -s` = en-openmpi* ]]; then
  echo "Setting up OpenCL/CUDA server environment"
	
  #------------------------------------------------------------------
	# Env for CUDA/OpenCL
  #------------------------------------------------------------------
	CUDA=/usr/local/cuda-7.0
	export PATH=$CUDA/bin:$PATH
	export LIBRARY_PATH=$CUDA/lib64:$LIBRARY_PATH
	export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH
	export CPATH=$CUDA/include:$CPATH

fi
