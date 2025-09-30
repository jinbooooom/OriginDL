wget https://arrayfire.s3.amazonaws.com/3.9.0/ArrayFire-v3.9.0_Linux_x86_64.sh  
sudo sh ArrayFire-v3.9.0_Linux_x86_64.sh  --skip-license --prefix=/opt/arrayfire 
export LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH 