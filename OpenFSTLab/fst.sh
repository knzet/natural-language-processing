#!/bin/bash

#sentence=$1

#echo "24 LATER KENNEDY ATE POPCORN 23" > test_file &&
echo $1 > test_file &&
farcompilestrings --symbols=wl.cmu --unknown_symbol="<unk>" test_file > test_cmu_file2.far &&
farextract test_cmu_file2.far &&
# where is test cmu file-1 coming from? how to control outfilename of farextract
fstcompose test_file-1 cmu.trans.fsm > cmustring.trans.fsm &&
fstproject --project_type=output cmustring.trans.fsm > output.far &&
farprintstrings --symbols=pl.cmu output.far
