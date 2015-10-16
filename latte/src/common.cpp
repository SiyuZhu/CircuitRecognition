#include "common.hpp"

namespace latte {
  Halide::Var x("x");
  Halide::Var y("y");
  Halide::Var c("c");
  Halide::Var n("n");

  Latte::Latte() {
    target_ = Halide::get_jit_target_from_environment();
    phase_ = Phase::TEST;
  }

  Latte *Latte::singleton_ = NULL;
}
