TARGET = ./val_filter

SRCS := $(wildcard ./src/*.cpp ./*.cpp)

OBJS := $(patsubst %cpp,%o,$(SRCS))

CFLG = -g -Wall -I/usr/local/Cellar/opencv/4.4.0_1/include/opencv4 -Iinc -I./ -std=c++11 `pkg-config --cflags --libs opencv4`


LDFG = -Wl, $(shell pkg-config --static opencv4 --cflags --libs ) 
CXX = g++

$(TARGET) : $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFG) 

%.o:%.cpp
	$(CXX) $(CFLG) -c $< -o $@ 

.PHONY : clean
clean:
	-rm ./*.o