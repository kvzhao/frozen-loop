python play_icegame.py -xL=32 -l=Models/CNN_prescale2_L2/ --num_tests=40000 -pp=s6 > log6 &
id1=$!
wait $id1
