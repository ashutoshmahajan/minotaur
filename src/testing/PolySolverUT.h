/* 
 *     MINOTAUR -- It's only 1/2 bull
 *
 *     (C)opyright 2009 - 2013 The MINOTAUR Team.
 */

#ifndef POLYSOLVERUT_H
#define POLYSOLVERUT_H

#include <string>

#include <cppunit/TestCase.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TextTestResult.h>
#include <cppunit/extensions/HelperMacros.h>

#include <AMPLInterface.h>
#include <Problem.h>

using namespace MINOTAUR_AMPL;


// read instance using ampl and test:
// Number of variables and their types,
// Number of constraints and their types,
// Function evaluations,
// Gradient evaluations,
// Hessian evaluations.

class PolySolverUT : public CppUnit::TestCase {

public:
  PolySolverUT(std::string name) : TestCase(name) {}
  PolySolverUT() {}

  void setUp();         
  void tearDown();   // need not implement

  void testSize();
  

  CPPUNIT_TEST_SUITE(PolySolverUT);
  CPPUNIT_TEST(testSize);

  CPPUNIT_TEST_SUITE_END();

private:
  AMPLInterfacePtr iface_;
  Minotaur::ProblemPtr inst_;
};

#endif

// Local Variables: 
// mode: c++ 
// eval: (c-set-style "k&r") 
// eval: (c-set-offset 'innamespace 0) 
// eval: (setq c-basic-offset 2) 
// eval: (setq fill-column 78) 
// eval: (auto-fill-mode 1) 
// eval: (setq column-number-mode 1) 
// eval: (setq indent-tabs-mode nil) 
// End:
