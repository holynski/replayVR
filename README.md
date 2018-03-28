=================================================

                  Replay VR

=================================================

# Dependencies

To run, first you must install the following packages:

TheiaSfM
GFlags
GLog
Eigen
CGAL
FFMPEG (libavcodec/libavutil/libavformat)
  -- For windows, see below.
ZLIB

# Compilation 

Then, to compile:

`mkdir build
cd build
cmake .. 
make -j4`

# FAQ: 

## How to compile ffmpeg + x264 using Visual Studio 2015+

1. Download "MSYS2 x86_64" from "http://msys2.github.io" and install into "C:\workspace\windows\msys64" 
2. Run: `pacman -S make gcc diffutils mingw-w64-{i686,x86_64}-pkg-config mingw-w64-i686-nasm mingw-w64-i686-yasm`
3. Rename C:\workspace\windows\msys64\usr\bin\link.exe to C:\workspace\windows\msys64\usr\bin\link_orig.exe, in order to use MSVC link.exe
4. Run "Visual Studio x64 Native Tools Command Prompt" or equivalent
5. Inside the command prompt, run: `C:\workspace\windows\msys64\msys2_shell.cmd -mingw64 -use-full-path`
6. In desired directory: `git clone http://git.videolan.org/git/x264.git`
7. `cd x264 && CC=cl ./configure --enable-static --prefix=/usr/local --disable-cli --disable-asm && make && make install`
8. Download sources from "http://www.ffmpeg.org/download.html"
9. `cd ffmpeg-x.x.x && export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig && ./configure --toolchain=msvc --arch=x86_64 --enable-yasm  --enable-asm --enable-shared --disable-static --disable-programs --enable-avresample --enable-libx264 --enable-gpl --prefix=./install && make && make install`
