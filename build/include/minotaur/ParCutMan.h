//
//     MINOTAUR -- It's only 1/2 bull
//
//     (C)opyright 2010 - 2017 The MINOTAUR Team.
//

/**
 * \file ParCutMan.h
 * \brief Manages storage, addition and deletion of cuts to problem.
 * \author Meenarli Sharma, Prashant Palkar, IIT Bombay
 */

#ifndef MINOTAURPARCUTMAN_H
#define MINOTAURPARCUTMAN_H

#include <list>
#include "CutManager.h"
#include "Types.h"


namespace Minotaur {

/**
 * \brief Derived class for managing cuts. Add and remove cuts based on
 * priority and violation.
 */
class ParCutMan : public CutManager {

public:
  /// Empty constructor.
  ParCutMan();

  /**
   * \brief Default constructor.
   *
   * \param [in] env Minotaur Environment pointer.
   * \param [in] p Problem pointer to which cuts will be added or deleted.
   */
  ParCutMan(EnvPtr env, ProblemPtr p);

  /// Destroy.
  ~ParCutMan();

  // Base class method.
  void addCut(CutPtr c);
  
  void addCutToPool(CutPtr cut);
  
  std::vector<ConstraintPtr > getPoolCons();

  ConstraintPtr addCut(ProblemPtr, FunctionPtr, double, double, 
                       bool, bool) {return ConstraintPtr();};

  // Base class method.
  void addCuts(CutVectorIter cbeg, CutVectorIter cend);

  // Base class method.
  UInt getNumCuts() const;

  // Base class method.
  UInt getNumEnabledCuts() const;

  // Base class method.
  UInt getNumDisabledCuts() const;

  // Base class method.
  UInt getNumNewCuts() const;
  
  // Base class method.
  void postSolveUpdate(ConstSolutionPtr sol, EngineStatus eng_status);

  // Base class method.
  void separate(ProblemPtr p, ConstSolutionPtr sol, bool *separated,
                UInt *n_added);

  // Base class method.
  void write(std::ostream &out) const;

  // Base class method.
  void writeStats(std::ostream &out) const;

private:

  /// Absolute tolerance limit for comparing double values.
  double absTol_;

  /// Cut storage for disabled cuts, i.e., those not added to the problem.
  CutList disCuts_;

  /// Cut storage for enabled cuts, i.e. those added to the problem.
  CutList enCuts_;

  /// Environment.
  EnvPtr env_;

  /// For logging.
  LoggerPtr logger_;

  /// Maximum number of iterations before which a disabled cut is deleted. 
  UInt maxDisCutAge_;

  /**
   * \brief Maximum number of iterations before which an inactive cut is moved out
   * of the problem.
   */
  UInt maxInactCutAge_;

  /// For logging.
  const static std::string me_;

  /**
   * Cut storage for new cuts, i.e. those sent to pool after previous
   * separate or postSolveUpdate().
   */
  CutList newCuts_;

  /// The relaxation problem that cuts are added to and deleted from.
  ProblemPtr p_;

  CutList pool_;

  void addToRel_(CutPtr cons, bool new_cut);

  void addToPool_(CutPtr cons);

};

//typedef ParCutMan * ParCutManPtr;
}
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
