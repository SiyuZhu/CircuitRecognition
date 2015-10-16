PROJ=gates
DATA_DIR=../../data
PROJ_DIR=$DATA_DIR/$PROJ

./cnn --pipeline_name="GATES_PIPELINE" \
    --param_files="${PROJ_DIR}/layer_params0.dat,${PROJ_DIR}/layer_params1.dat,${PROJ_DIR}/layer_params2.dat,${PROJ_DIR}/layer_params3.dat"

PROJ=mnist
PROJ_DIR=$DATA_DIR/$PROJ

./cnn --pipeline_name="MNIST_PIPELINE" \
    --param_files="${PROJ_DIR}/layer_params0.dat,${PROJ_DIR}/layer_params1.dat,${PROJ_DIR}/layer_params2.dat,${PROJ_DIR}/layer_params3.dat"