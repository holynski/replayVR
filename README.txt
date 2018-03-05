=================================================

                  Replay VR

=================================================

To run, first you must install the following packages:

TheiaSfM
GFlags
GLog
Eigen
CGAL
FFMPEG (libavcodec/libavutil/libavformat)
  -- For windows, see below.
ZLIB



Then, to compile:

mkdir build
cd build
cmake .. 
make -j4



##### How to compile ffmpeg + x264 using Visual Studio 2015 #####
##### Building this way will make the DLLs compatible with SEH, so there will be no need to use /SAFESEH:NO when compiling your code #####

##### SOURCES:
### https://gist.github.com/RangelReale/3e6392289d8ba1a52b6e70cdd7e10282
### https://pracucci.com/compile-ffmpeg-on-windows-with-visual-studio-compiler.html
### https://gist.github.com/sailfish009/8d6761474f87c074703e187a2bc90bbc
### http://roxlu.com/2016/057/compiling-x264-on-windows-with-msvc

* Download "MSYS2 x86_64" from "http://msys2.github.io" and install into "C:\workspace\windows\msys64" 

# pacman -S make gcc diffutils mingw-w64-{i686,x86_64}-pkg-config mingw-w64-i686-nasm mingw-w64-i686-yasm

* Rename C:\workspace\windows\msys64\usr\bin\link.exe to C:\workspace\windows\msys64\usr\bin\link_orig.exe, in order to use MSVC link.exe

===== 32 BITS BEGIN
* Run "VS2015 x86 Native Tools Command Prompt"

* Inside the command prompt, run: 
# C:\workspace\windows\msys64\msys2_shell.cmd -mingw32 -use-full-path
===== 32 BITS END

===== 64 BITS BEGIN
* Run "VS2015 x64 Native Tools Command Prompt"

* Inside the command prompt, run: 
# C:\workspace\windows\msys64\msys2_shell.cmd -mingw64 -use-full-path
===== 64 BITS END

### x264 ###
# git clone http://git.videolan.org/git/x264.git
# cd x264
# CC=cl ./configure --enable-static --prefix=/usr/local --disable-cli --disable-asm
# make
# make install

### ffmpeg ###
* Download sources from "http://www.ffmpeg.org/download.html"
# cd ffmpeg-3.3.2
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
===== 32 BITS BEGIN
# ./configure --toolchain=msvc --arch=x86 --enable-yasm  --enable-asm --enable-shared --disable-static --disable-programs --enable-avresample --enable-libx264 --enable-gpl --prefix=./install
===== 32 BITS END
===== 64 BITS BEGIN
# ./configure --toolchain=msvc --arch=x86_64 --enable-yasm  --enable-asm --enable-shared --disable-static --disable-programs --enable-avresample --enable-libx264 --enable-gpl --prefix=./install
===== 64 BITS END
# make
# make install