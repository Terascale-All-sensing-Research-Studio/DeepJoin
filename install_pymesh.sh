# Make libs
if [ ! -d "libs" ] 
then
    mkdir libs
fi
cd libs &&

pip install cmake

# Install CGAL
sudo apt-get install -y \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    python3-dev \
    libcgal-dev

# Clone and install
git clone https://github.com/PyMesh/PyMesh.git && \
cd PyMesh && \
git submodule update --init && \
./setup.py build && \
./setup.py install
