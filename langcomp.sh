# compile if needed
# g++ helpervectmat.cpp logfcts.cpp main.cpp -g -o app -O2 \
#     -I ./src/lib/eigen

# run of c++ code
echo "-------Run of C++ code--------"
time ./app
python visu.py
xdg-open ./cpppost.png

# run of python code
echo "-------Run of python code-------"
time python main.py
xdg-open ./pythonpost.png
