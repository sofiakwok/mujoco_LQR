
#MAC
#COMMON=-O2 -I../../include -L../../bin -mavx -pthread
#LIBS = -w -lmujoco200 -lglfw.3
#CC = gcc

#LINUX
COMMON=-O2 -I../../include -L../lib -std=c++17 -pthread -Wl,-no-as-needed -Wl,-rpath,'$$ORIGIN'/../lib
#LIBS = -lmujoco200 -lGL -lm -lglew ../../bin/libglfw.so.3
LIBS = -lGL -lm -lglfw ../../lib/libmujoco.so
CC = gcc

#WINDOWS
#COMMON=/O2 /MT /EHsc /arch:AVX /I../../include /Fe../../bin/
#LIBS = ../../bin/glfw3.lib  ../../bin/mujoco200.lib
#CC = cl

ROOT = LQR

all:
	$(CC) $(COMMON) LQR_hopper.cc $(LIBS) -o ../../bin/$(ROOT)

main.o:
	$(CC) $(COMMON) -c LQR_hopper.cc

clean:
	rm *.o ../../bin/$(ROOT)
