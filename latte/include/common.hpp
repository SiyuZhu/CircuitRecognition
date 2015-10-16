#ifndef COMMON_HPP
#define COMMON_HPP

#include "Halide.h"

namespace latte {
  extern Halide::Var x;
  extern Halide::Var y;
  extern Halide::Var c;
  extern Halide::Var n;

  class Latte {
  public:
    enum Phase {TRAIN, TEST};
  private:
    static Latte *singleton_;
    Halide::Target target_;
    Phase phase_; 
    
    Latte();
  public:
    inline static Latte* singleton() {
      if(singleton_ == NULL)
	singleton_ = new Latte();
      return singleton_;
    }
    inline void set_target(Halide::Target target) {target_ = target;}
    inline void set_phase(Phase phase) {phase_ = phase;}
    inline Halide::Target target() {return target_;}
    inline Phase phase() {return phase_;}
  };
}

#endif
