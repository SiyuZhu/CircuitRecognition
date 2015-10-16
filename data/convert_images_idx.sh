GATE_ROOT=./gates
TOOL_ROOT=./tools
VAL_DATA=$GATE_ROOT/test_data
VAL_TXT=$VAL_DATA/test.txt
IMG_IDX=$GATE_ROOT/images.idx
LBL_IDX=$GATE_ROOT/labels.idx

$TOOL_ROOT/convert_images_idx --valdir_arg=$VAL_DATA --valtxt_arg=$VAL_TXT --imgidx_arg=$IMG_IDX --lblidx_arg=$LBL_IDX
#$TOOL_ROOT/testCompression --imgidx_arg=$IMG_IDX