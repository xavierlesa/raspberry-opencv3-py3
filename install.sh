#!/bin/bash

echo
echo "Instalación raspi + opencv 4.0.1 + python 3"
echo

opencv3_version=3.4.5
opencv4_version=4.0.1

pyenv="opencv_py3"

PS3='Elegí el tipo de instalación: '
options=($opencv3_version $opencv4_version otra salir)
select opt in "${options[@]}"
do
    case $opt in
        $opencv3_version)
            opencv_version=$opencv3_version
            break
            ;;
        $opencv4_version)
            opencv_version=$opencv4_version
            break
            ;;
        otra)
            echo "Indicar que versión instalar (ej: 3.4.1) "
            read opencv_version
            break
            ;;
        salir)
            break
            ;;
        *) echo "Opción invalida $REPLY";;
    esac
done

echo "Instalando OpenCV $opencv_version ... "

# download sources
if ! test -d src
then
    echo "Crea ./src ... "
    mkdir -p ./src
else
    echo "Accediedo a ./src ... "
fi

cd src

wget -c https://github.com/opencv/opencv/archive/$opencv_version.tar.gz -O opencv-$opencv_version.tar.gz
tar -xvf opencv-$opencv_version.tar.gz

wget -c https://github.com/opencv/opencv_contrib/archive/$opencv_version.tar.gz -O opencv_contrib-$opencv_version.tar.gz
tar -xvf opencv_contrib-$opencv_version.tar.gz


# Install pip
sudo apt-get install python3-pip

# Install virtualenv
pip3 install virtualenv virtualenvwrapper

if [ $WORKON_HOME == '' ]; then

    if [ -f "$HOME/.bashrc" ]; then
        profile_file="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        profile_file="$HOME/.bash_profile"
    fi

    echo "" >> $profile_file
    echo "# virtualenv and virtualenvwrapper" >> $profile_file
    echo "export WORKON_HOME=$HOME/.virtualenvs" >> $profile_file
    echo "source /usr/local/bin/virtualenvwrapper.sh" >> $profile_file
fi


# Create env
# this command set a new env after created
# new prompt will be like "(opencv_py3) pi@raspberrypi"

# check if not exists first
if [ $(lsvirtualenv -b | grep $pyenv) ]; then
    workon $pyenv
else
    mkvirtualenv $pyenv -p python3
    setvirtualenvproyect
fi

# Install python dependencies with pip because we are into env of python3
pip install numpy

# Go to opencv/build
cd opencv-$opencv_version
if ! test -d build
then
    echo "Crea ./build ... "
    mkdir -p ./build
else
    echo "Accediedo a ./build ... "
fi

cd ./build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-$opencv_version/modules \
    -D WITH_CUDA=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS= OFF ..


make -j4

sudo make install
sudo ldconfig

#cd $(python -c "import sys; print([i for i in sys.path if i.startswith('/usr/lib')][0])")/site-packages/


