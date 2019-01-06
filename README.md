# raspberry-opencv3-py3
Notas para instalar OpenCV3 sobre Python 3 en rapsberry


# 1- Instalar dependencias

```
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
```

Si todo salió bien es necesario hacer un `reboot` para continuar con el `upgrade`.

Ahora instalamos los `essential` para poder compilar:

```
sudo apt-get install build-essential git cmake pkg-config
```

Instalamos las dependencias para manejo de imagenes:

```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
```

**Nota**: a mi no me funcionó instalar `libpng12-dev` no sé si es por la versión del OS o que, pero lo pasé por alto, entonces si fallá instalar solo los demás.

```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev
```

Lo mismo pero para el manejo de compresión de videos:

```
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
```

**Opcional**: si fuese necesario se puede instalar GTK para usar la UI del OS para ver en tiempo real lo que estamos procesando en OpenCV. Para esto hacemos:

```
sudo apt-get install libgtk2.0-dev
```

Un pequeña libreria para el manejo optimizado de matrices (matemáticas) para OpenCV:

```
sudo apt-get install libatlas-base-dev gfortran
```

Instalamos los Header de Python[2,3], aquí si solo vamos a usar Python 2 no es necesario el otro y viceversa:

```
sudo apt-get install python2.7-dev python3-dev
```


# 2 - Descargamos OpenCV

Hay que buscar [aquí](https://github.com/opencv/opencv/releases) el release que se quiera instalar, yo instalé `3.1.0` pero funcionaría igual para otra versión como la `3.4.3` que es la que es la versión actual `stable`.

```
cd ~
wget -c https://github.com/opencv/opencv/archive/3.4.3.tar.gz -O opencv-3.4.3.tar.gz
tar -xvf opencv-3.4.3.tar.gz
```

Descargamos los `contribs` de OpenCV para tener funcionalidades extras. Las versiones de OpenCV y los contrib deben ser las mismas.

```
cd ~
wget -c https://github.com/opencv/opencv_contrib/archive/3.4.3.tar.gz -O opencv_contrib-3.4.3.tar.gz
tar -xvf opencv_contrib-3.4.3.tar.gz
```


# 3 - Setup de Python

Instalamos `pip` 

```
sudo apt-get install python-pip
```

Y `virtualenv`

```
pip install virtualenv virtualenvwrapper
```

Actualizamo el `ENV` para que levante el `command` de virtualenv, para esto editar `~/.bash_profile` o `~/.profile` dependiendo de la versión de OS que tengas, y le agregas esto al final:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

Ahora podes hacer `workon ...` y te activa el `env`.

Creamos el env sobre python3:

```
mkvirtualenv opencv_py3 -p python3
```

Creamos un directorio donde vamos a trabajar, y activamos el env:

```
mkdir ~/src
cd ~/src
workon opencv_py3
# cuando activas con workon cambia el prompt agregando al inicio el nombre del env entre parentesis
# aquí sería (opencv_py3)
(opencv_py3) setvirtualenvproject # setea ~/src como el dir de trabajo para el env 
```

Finalmente instalamos `numpy`

```
(opencv_py3) pip install numpy
```

# 4 - Compilar OpenCV

Volvemos al src que desacargamos de opencv `cd ~/opencv-3.4.3` y vamos a hacer el `build` del source:

```
(opencv_py3) cd ~/opencv-3.4.3/
(opencv_py3) mkdir build
(opencv_py3) cd build
(opencv_py3) cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.3/modules \
	-D BUILD_EXAMPLES=ON ..
```

Aquí si tenes errores puede ser por una de dos cosas:


  - Un bug con `cmake` y los exmaples y se arregla con `-D INSTALL_C_EXAMPLES=OFF`

  - Un bug en `OpenCV` que se arregla con `-D ENABLE_PRECOMPILED_HEADERS=OFF ...`
  

Una alternativa para cuando estas con poco espacio es solo compilar lo que realmente vas a usar:

```
(opencv_py3) cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D BUILD_opencv_java=OFF \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=ON \
-D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.3/modules \
-D WITH_CUDA=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS= OFF ..
```

 
Esto va a tomar un rato y luego si todo salió bien podemos hacer `make`

```
(opencv_py3) make -j4
```

Si `-j4` te da errores proba sin setear los nucleos haciendo `make` a cecas.

```
(opencv_py3) make clean # si, borramos el build fallido
(opencv_py3) make
```

Y si todo salió bien podemos instalar y cargar el `module`:

```
(opencv_py3) sudo make install
(opencv_py3) sudo ldconfig
```

A este punto ya esta todo instalado, ahora solo nos queda linkear el `module` a nuestro `env`:

```
(opencv_py3) cd /usr/local/lib/python3.5/site-packages/
(opencv_py3) sudo mv cv2.cpython-35m-arm-linux-gnueabihf.so cv2.so

# Linke a env de python3
(opencv_py3) cd ~/.virtualenvs/opencv_py3/lib/python3.5/site-packages/
(opencv_py3) ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so
```

Para probar si todo fue bien importamos `cv2` y probamos la versión:

```
(opencv_py3) pip install ipython # opcional

(opencv_py3) ipython # o solo python si no instalas ipython

In [1]: import cv2
In [2]: cv2.__version__
Out[2]: '3.4.3'
```

DONE!
