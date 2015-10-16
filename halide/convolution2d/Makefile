HFLAGS=-lHalide -lpng -lpthread -ldl
TARGET=conv

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	g++ $< -o $@ $(HFLAGS)

clean:
	rm -f $(TARGET) *.dat
