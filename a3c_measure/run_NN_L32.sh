python play_icegame.py -xL=32 -l=Models/NN_prescale2_L2 -p=simple --num_tests=60000 -pp=s5 > log5 &
id1=$!
wait $id1
