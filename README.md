# Installation
```sh
python3 -m virtualenv -p python3 .env
source .env/bin/activate
pip3 install -r requirements.txt
(mkdir -p resources; cd resources; curl http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | bzip2 -d > shape_predictor_68_face_landmarks.dat)

```

# Running
```sh
source .env/bin/activate

# Glasses filter
./main.py --filter glasses --footage glasses.png
# Moustache filter
./main.py --filter moustache --footage resources/moustache.png

```

# Research paper
Paper written on this is available here: https://me.syzible.com/snapchat-filters.pdf

3 filters for CS7434 augmented reality - face swap, glasses and moustache

Clone the repo and create a directory in it called "resources". In this, you need the pre-trained face data available here:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

In resources, place any images you want to use in it for filters (face swap images, moustache image, glasses image, etc); modify the code as appropriate.

Make sure you have Python 3 installed, see here for easy installation with Brew on OSX http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/


If you want to create bug fixes or extend functionality, feel free to send pull requests.
