# g++ helpervectmat.cpp main.cpp -o app -std=c++11 -O2 

g++ helpervectmat.cpp logfcts.cpp main.cpp -g -o app -O2 \
    -I ../src/lib/eigen

./app

# python visu.py