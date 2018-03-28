=================================================

#Replay VR

=================================================

## Dependencies

To run, first you must install the following packages. 

The first few can be installed quite easily using Homebrew on OSX, your package manager of choice on Linux, or the CMake GUI on Windows:
* GFlags
* GLog
* Eigen
* CGAL
* ZLIB

The last one is a bit different:

FFMPEG (libavcodec/libavutil/libavformat)
Since this library depends on features of FFMPEG which are currently not on the main branch, you must first clone the FFMPEG repository, replace a couple files, and then install on your system using CMake. In order to install, you must:
1. `git clone https://github.com/FFmpeg/FFmpeg`
2. Replace the files `FFmpeg/libavformat/mov.c`, `FFmpeg/libavutil/spherical.h`, and `FFmpeg/libavutil/spherical.c` with the ones provided in the ffmpeg_mods folder.
3. If compiling on Windows, skip below to the FAQ for compilation instructions, otherwise compile and install with `mkdir build && cd build && cmake .. && make -j && make install`

## Compilation 

Then, to compile:

`mkdir build`

`cd build`

`cmake ..`

`make -j`

## FAQ: 

### How to compile ffmpeg + x264 using Visual Studio 2015+

1. Download "MSYS2 x86_64" from "http://msys2.github.io" and install into "C:\workspace\windows\msys64" 
2. Run: `pacman -S make gcc diffutils mingw-w64-{i686,x86_64}-pkg-config mingw-w64-i686-nasm mingw-w64-i686-yasm`
3. Rename `C:\workspace\windows\msys64\usr\bin\link.exe` to `C:\workspace\windows\msys64\usr\bin\link_orig.exe`, in order to use MSVC link.exe
4. Run "Visual Studio x64 Native Tools Command Prompt" or equivalent. This will ensure that all the Visual Studio path variables are used.
5. Inside the command prompt, run: `C:\workspace\windows\msys64\msys2_shell.cmd -mingw64 -use-full-path`
6. In desired directory: `git clone http://git.videolan.org/git/x264.git`
7. `cd x264 && CC=cl ./configure --enable-static --prefix=/usr/local --disable-cli --disable-asm && make && make install`
8. Download sources from "http://www.ffmpeg.org/download.html"
9. `cd ffmpeg-x.x.x && export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig && ./configure --toolchain=msvc --arch=x86_64 --enable-yasm  --enable-asm --enable-shared --disable-static --disable-programs --enable-avresample --enable-libx264 --enable-gpl --prefix=./install && make && make install`
