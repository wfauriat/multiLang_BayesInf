# g++ helpervectmat.cpp main.cpp -o app -std=c++11 -O2 

g++ helpervectmat.cpp main.cpp -o app -O2 \
    -I ./src/lib/eigen

./app 50