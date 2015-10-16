CVFLAGS = `pkg-config --cflags --libs opencv`
TARGET=main

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	g++ $< -o $@ $(CVFLAGS)

clean:
	rm -f $(TARGET)
