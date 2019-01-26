python play_icegame.py -l=Models/NN_prescale2_L2 -p=simple --num_tests=60000 -pp=s8 > log8 &
id1=$!
wait $id1
