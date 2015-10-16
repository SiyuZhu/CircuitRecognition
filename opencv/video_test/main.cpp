#include <highgui.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main (int argc, char** argv) {
  if (argc < 2) {
    fprintf (stderr, "Require video file as argument\n");
    exit(1);
  }

  VideoCapture vc;
  vc.open(argv[1]);
  if (!vc.isOpened()) {
    fprintf (stderr, "Cannot open %s\n", argv[1]);
    exit(1);
  }

  // data structures to access frames
  Mat frame;
  int nframes = 0;
  
  // jpg params
  vector<int> params;
  params.push_back(CV_IMWRITE_JPEG_QUALITY);
  params.push_back(100);

  while (vc.read(frame)) {
    // write first few frames
    if (nframes <= 2) {
      char name[20];
      sprintf (name, "frame%d.jpg", nframes);
      imwrite(name, frame, params);
    }
    nframes++;
  };

  printf ("There are %d frames in this video\n", nframes);
}
