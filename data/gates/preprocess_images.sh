DIR="$( cd "$(dirname "$0")" ; pwd -P )"
TOOL_ROOT=$DIR/../tools
TRAIN_TEST_DATA=$DIR/train_test_data
IMG_DIRS=$TRAIN_TEST_DATA/AND,$TRAIN_TEST_DATA/NAND,$TRAIN_TEST_DATA/NOT,$TRAIN_TEST_DATA/NOR,$TRAIN_TEST_DATA/OR,$TRAIN_TEST_DATA/XOR
TRAIN_DATA=$DIR/train_data
TRAIN_TXT=$TRAIN_DATA/train.txt
TEST_DATA=$DIR/test_data
TEST_TXT=$TEST_DATA/test.txt
IMG_DIM=28

make -C $TOOL_ROOT all

# Image normalization
$TOOL_ROOT/rename_images --imgdir_args=$IMG_DIRS
$TOOL_ROOT/resize_images --imgdir_args=$IMG_DIRS --img_width_arg=76 --img_height_arg=50
$TOOL_ROOT/center_images --imgdir_args=$IMG_DIRS
# Resize images to 28*28 - faster training and more accurate model
$TOOL_ROOT/resize_images --imgdir_args=$IMG_DIRS --img_width_arg=$IMG_DIM --img_height_arg=$IMG_DIM
$TOOL_ROOT/augment_images --imgdir_args=$IMG_DIRS --aug8_args
# Distributes images between training and testing
$TOOL_ROOT/allocate_images --imgdir_args=$IMG_DIRS --traindir_arg=$TRAIN_DATA \
			   --traintxt_arg=$TRAIN_TXT --valdir_arg=$TEST_DATA \
			   --valtxt_arg=$TEST_TXT 
