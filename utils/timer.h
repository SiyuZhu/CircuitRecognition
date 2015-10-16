//---------------------------------------------------------
// Timer.h
//---------------------------------------------------------
#ifndef __TIMER_H__
#define __TIMER_H__
#include <time.h>
#include <string.h>
#include <stdio.h>

#define TIMER_ON


//---------------------------------------------------------
// Timer is an object which helps profile programs using
// the clock() function.
// - By default, a timer is stopped when you instantiate it
//   and must be started manually
// - Passing True to the constructor starts the timer when
//   it is constructed
// - When the timer is destructed it prints stats to stdout
//---------------------------------------------------------
class Timer {

  #ifdef TIMER_ON    
    char binName[50];
    unsigned startTime, nCalls;
    float totalTime;
    
    public:
    
    Timer (const char* Name, bool On=false) {
      if (On) {
        startTime = clock();
        nCalls = 1;
      }
      else {
        startTime = 0;
        nCalls = 0;
      }
      totalTime = 0;	
      strcpy(binName, Name);
    }

    ~Timer () {
        if (nCalls > 0) {
            printf ("%-30s: ", binName);
            printf ("%6d calls; ", nCalls);
            printf ("%7.3f secs total time; ", totalTime);
            printf ("%7.4f msecs average time;\n", 1000*totalTime/nCalls);
        }
    }
    
    void start() {
        startTime = clock();
        nCalls++;
    }
    
    void stop() {
        totalTime += float(clock() - startTime)/CLOCKS_PER_SEC;
    }
  #else
    public:
    Timer (const char* Name, bool On=true) {}
    void start() {}
    void stop() {}
  #endif
};

#endif
