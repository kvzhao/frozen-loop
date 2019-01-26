python play_icegame.py -l=Models/CNN_LargeLoop/ --num_tests=20000 -pp=s6 > log6 & 
id1=$!
wait $id1
