#ifndef LATTE_INTERNAL_THREAD_HPP_
#define LATTE_INTERNAL_THREAD_HPP_

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace latte {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread {
 public:
  InternalThread();
  virtual ~InternalThread();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExit();

  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry();

  boost::shared_ptr<boost::thread> thread_;
};

}  // namespace latte

#endif  // LATTE_INTERNAL_THREAD_HPP_
