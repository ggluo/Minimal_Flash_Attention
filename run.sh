mkdir build
cd build
cmake ..
make
cd ..

for((i=0;i<=7;i++))
do
    echo -n "${i}..."
    ./flash ${i}
done