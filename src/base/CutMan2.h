//
//     MINOTAUR -- It's only 1/2 bull
//
//     (C)opyright 2010 - 2013 The MINOTAUR Team.
//

/**
 * \file CutMan2.h
 * \brief Manages addition and deletion of cuts to problem.
 * \author Mahdi Hamzeei, University of Wisconsin-Madison
 */

#ifndef MINOTAURCUTMAN2_H
#define MINOTAURCUTMAN2_H

#include <list>
#include <map>
#include "CutManager.h"

namespace Minotaur {

class Constraint;

class Node;
class Timer;
typedef boost::shared_ptr<Constraint> ConstraintPtr;
typedef boost::shared_ptr<Node> NodePtr;
typedef std::list<CutPtr> cutList;

/**
 * The CutManager class is meant to manage the cutting planes generated by
 * different cut generators and handlers. 
 */

struct CutStat {
  Int numAddedCuts;
  Int numDeletedCuts;
  Int numPoolToRel;
  Int numRelToPool;
  Int callNums;
  Int PoolSize;
  Int RelSize;
  Int numCuts;
  };

class CutMan2 : public CutManager {

public:
  /// Default constructor.
  CutMan2();

  /// Constructor that loads the relaxation problem. 
  CutMan2(EnvPtr env,ProblemPtr p);

  /// Destroy.
  ~CutMan2();

  // base class method
  void addCut(CutPtr c);

  // base class method
  void addCuts(CutVectorIter cbeg, CutVectorIter cend);

  UInt getNumCuts() const { return numCuts_; };

  UInt getNumEnabledCuts() const { return rel_.size(); };

  UInt getNumDisabledCuts() const { return pool_.size(); };

  UInt getNumNewCuts() const { return 0;};

  // base class method
  ConstraintPtr addCut(ProblemPtr rel,FunctionPtr fn, Double lb, Double ub, 
		Bool directToRel, Bool neverDelete);

  // base class method
  void NodeIsBranched(NodePtr node, ConstSolutionPtr sol, Int num);//, ConstSolutionPtr sol);

  // base class method
  void NodeIsProcessed(NodePtr node);

  // base class method
  void postSolveUpdate(ConstSolutionPtr , EngineStatus ) {};

  // base class method
  void separate(ConstSolutionPtr, Bool*, UInt*) {};

  // base class method
  void updatePool(ProblemPtr rel, ConstSolutionPtr sol);

  // base class method
  void updateRel(ConstSolutionPtr sol, ProblemPtr rel);

  // base class method
  void write(std::ostream &out) const;
  
  // base class method
  void writeStats(std::ostream &out) const;

  void writeStat();

  struct ctMngrInfo{
    Double t;
    Int RelSize;
    Int PoolSize;
    Int RelToPool;
    Int PoolToRel;
    Double RelAve;
    Double PoolAve;
    Int RelTr;
    Int PoolTr;
    Int PrntActCnt;
  };

  ctMngrInfo getInfo() {return ctmngrInfo_;}

private:
  /// Cut pool
  cutList pool_;

  /// list of cuts in the relaxation
  cutList rel_;

  /// Map of active cuts to a node
  std::map< NodePtr , cutList > NodeCutsMap_;

  /// Map of number of children to a node
  std::map< NodePtr , Int > ChildNum_;

  /// Environment.
  EnvPtr env_;

  /// Random vector to check for duplicacy.
  Double* hashVec_;

  /// For logging.
  LoggerPtr logger_;

  /// For logging.
  const static std::string me_;

  /// The relaxation problem that cuts are added to and deleted from.
  ProblemPtr p_;

  /// Adding cut to the relaxation
  void addToRel_(ProblemPtr rel, CutPtr cut, Bool newcut);

  /// Adding cut to the relaxation
  void addToRel_(CutPtr cut);
 
  /// Adding cut to the cut pool
  void addToPool_(CutPtr cut);

  /// Absolute tolerance
  Double absTol_;

  /**
   * \brief Maximum number of iterations before which an inactive cut is moved out
   * of the problem.
   */
  UInt MaxInactiveInRel_;
  
  /// Maximum number of iterations before which a cut in the pool is deleted.
  UInt MaxUnviolInPool_;

  /// Maximum pool size
  UInt PoolSize_;

  /// CutMan activation threshod
  UInt CtThrsh_;

  CutStat *stats_;
  
  Timer *timer_;

  /// Total time spent in cut manager
  Double ctMngrtime_;

  /// Time spent in updateCuts
  Double updateTime_;

  /// Time spent in checkCuts
  Double checkTime_;

  /// Time spent in NodeIsProcessed
  Double processedTime_;

  /// Time spent in NodeIsBranched
  Double branchedTime_;

  ctMngrInfo ctmngrInfo_;

  /// Maximum number of active children for a node to removing its active cuts
  Int PrntCntThrsh_;

  UInt numCuts_;


};
  typedef boost::shared_ptr <CutMan2> CutMan2Ptr;
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
