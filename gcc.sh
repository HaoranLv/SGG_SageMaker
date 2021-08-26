cd /usr/local/src/
sudo wget http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.xz
sudo tar xvf gcc-7.3.0.tar.xz
cd gcc-7.3.0/
sudo ./contrib/download_prerequisites
sudo ./configure -enable-checking=release -enable-languages=c,c++ -disable-multilib
sudo make -j4
ls /usr/local/bin | grep gcc
sudo make install
cd ~
find /usr/local/src/gcc-7.3.0/ -name "libstdc++.so*"
cd /usr/lib64
sudo cp /usr/local/src/gcc-7.3.0/stage1-x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6.0.24 .
sudo mv libstdc++.so.6 libstdc++.so.6.old
sudo ln -sv libstdc++.so.6.0.24 libstdc++.so.6