// 
//     MINOTAUR -- It's only 1/2 bull
// 
//     (C)opyright 2009 - 2013 The MINOTAUR Team.
// 

/**
 * \file BndProcessor.cpp
 * \brief Implement simple node-processor for branch-and-bound
 * \author Ashutosh Mahajan, IIT Bombay
 */
#include <cmath> // for INFINITY

#include "MinotaurConfig.h"
#include "Brancher.h"
#include "Engine.h"
#include "Environment.h"
#include "Handler.h"
#include "BndProcessor.h"
#include "Logger.h"
#include "Node.h"
#include "Option.h"
#include "Modification.h"
#include "Relaxation.h"
#include "SolutionPool.h"

using namespace Minotaur;

//#define SPEW 1

const std::string BndProcessor::me_ = "BndProcessor: ";

BndProcessor::BndProcessor()
  : contOnErr_(false),
    cutOff_(INFINITY),
    engine_(EnginePtr()),
    engineStatus_(EngineUnknownStatus),
    numSolutions_(0),
    relaxation_(RelaxationPtr()),
    ws_(WarmStartPtr())
{
  handlers_.clear();
  logger_ = (LoggerPtr) new Logger(LogInfo);
  stats_.inf = 0;
  stats_.opt = 0;
  stats_.prob = 0;
  stats_.proc = 0;
  stats_.ub = 0;
}


BndProcessor::BndProcessor (EnvPtr env, EnginePtr engine,
                            HandlerVector handlers)
  : contOnErr_(false),
    engine_(engine),
    engineStatus_(EngineUnknownStatus),
    numSolutions_(0),
    relaxation_(RelaxationPtr()),
    ws_(WarmStartPtr())
{
  cutOff_ = env->getOptions()->findDouble("obj_cut_off")->getValue();
  handlers_ = handlers;
  logger_ = (LoggerPtr) new Logger((LogLevel)env->getOptions()->
                                   findInt("node_processor_log_level")->
                                   getValue());
  stats_.bra = 0;
  stats_.inf = 0;
  stats_.opt = 0;
  stats_.prob = 0;
  stats_.proc = 0;
  stats_.ub = 0;
}


BndProcessor::~BndProcessor()
{
  handlers_.clear();
  logger_.reset();
  engine_.reset();
}


Bool BndProcessor::foundNewSolution()
{
  return (numSolutions_ > 0);
}


Branches BndProcessor::getBranches()
{
  ++stats_.bra;
  return branches_;
}


WarmStartPtr BndProcessor::getWarmStart()
{
  return ws_;
}


Bool BndProcessor::isFeasible_(NodePtr node, ConstSolutionPtr sol, 
                              SolutionPoolPtr s_pool, Bool &should_prune)
{
  should_prune = false;
  Bool is_feas = true;
  HandlerIterator h;
  // visit each handler and check feasibility. Stop on the first
  // infeasibility.
  for (h = handlers_.begin(); h != handlers_.end(); ++h) {
    is_feas = (*h)->isFeasible(sol, relaxation_, should_prune);
    if (is_feas == false || should_prune == true) {
      break;
    }
  }

  if (is_feas == true && h==handlers_.end()) {
    s_pool->addSolution(sol);
    ++numSolutions_;
    node->setStatus(NodeOptimal);
    ++stats_.opt;
    should_prune = true;
  }
  return is_feas;
}


void BndProcessor::process(NodePtr node, RelaxationPtr rel,
                          SolutionPoolPtr s_pool)
{
  Bool should_prune = true;
  Bool should_resolve;
  BrancherStatus br_status;
  ConstSolutionPtr sol;
  ModificationPtr mod;
  Int iter = 0;

  ++stats_.proc;
  relaxation_ = rel;

#if 0
  Double *svar = new Double[20];
  Bool xfeas = true;
  svar[1-1]   = 0.000000000  ;
  svar[2-1]   = 0.000000000  ;
  svar[3-1]   = 1.042899924  ;
  svar[4-1]   = 0.000000000  ;
  svar[5-1]   = 0.000000000  ;
  svar[6-1]   = 0.000000000  ;
  svar[7-1]   = 0.000000000  ;
  svar[8-1]   = 0.000000000  ;
  svar[9-1]   = 0.000000000  ;
  svar[10-1]  = 0.000000000  ;
  svar[11-1]  = 1.746743790  ;
  svar[12-1]  = 0.000000000  ;
  svar[13-1]  = 0.431470884  ;
  svar[14-1]  = 0.000000000  ;
  svar[15-1]  = 0.000000000  ;
  svar[16-1]  = 4.433050274  ;
  svar[17-1]  = 0.000000000  ;
  svar[18-1]  = 15.858931758 ;
  svar[19-1]  = 0.000000000  ;
  svar[20-1]  = 16.486903370 ;

  for (UInt i=0; i<20; ++i) {
    if (svar[i] > rel->getVariable(i)->getUb()+1e-6 || 
        svar[i] < rel->getVariable(i)->getLb()-1e-6) {
      xfeas = false;
      break;
    }
  }
  if (true==xfeas) {
    std::cout << "xsol feasible in node " << node->getId() << std::endl;
  } else {
    std::cout << "xsol NOT feasible in node " << node->getId() << std::endl;
  }
#endif 

  // loop for branching and resolving if necessary.

  while (true) {
    ++iter;
    should_resolve = false;

#if SPEW
  logger_->MsgStream(LogDebug) << me_ << "iteration " << iter << std::endl;
#endif

    //relaxation_->write(std::cout);
    solveRelaxation_();

    sol = engine_->getSolution();

    // check if the relaxation is infeasible or if the cost is too high.
    // In either case we can prune. Also set lb of node.
    should_prune = shouldPrune_(node, sol->getObjValue(), s_pool);
    if (should_prune) {
      break;
    }

    // update pseudo-cost from last branching.
    if (iter == 1) {
      brancher_->updateAfterLP(node, sol);
    }

    // check feasibility. if it is feasible, we can still prune this node.
    isFeasible_(node, sol, s_pool, should_prune);
    if (should_prune) {
      break;
    }


    //relaxation_->write(std::cout);

    //save warm start information before branching. This step is expensive.
    ws_ = engine_->getWarmStartCopy();
    branches_ = brancher_->findBranches(relaxation_, node, sol, s_pool, 
                                        br_status, mod);
    if (br_status==PrunedByBrancher) {

      should_prune = true;
      node->setStatus(NodeInfeasible);
      stats_.inf++;
      break;
    } else if (br_status==ModifiedByBrancher) {
      node->addModification(mod);
      mod->applyToProblem(relaxation_);
      if (should_prune) {
        break;
      }
      should_resolve = true;
    } 
    if (should_resolve == false) {
      break;
    }
  }
#if 0
  if ((true==should_prune || node->getLb() >-4150) && true==xfeas) {
    std::cout << "problem here!\n";
    std::cout << node->getStatus() << "\n";
    rel->write(std::cout);
    exit(0);
  }
#endif

  return;
}


Bool BndProcessor::shouldPrune_(NodePtr node, Double solval, 
                               SolutionPoolPtr s_pool)
{
  Bool should_prune = false;
#if SPEW
  logger_->MsgStream(LogDebug2) << me_ << "solution value = " << solval
                                << std::endl; 
#endif
  switch (engineStatus_) {
   case (FailedInfeas):
     logger_->MsgStream(LogInfo) << me_ << "failed to converge "
                                 << "(infeasible) in node " << node->getId()
                                 << std::endl;
     node->setStatus(NodeInfeasible);
     should_prune = true;
     ++stats_.inf;
     ++stats_.prob;
     break;
   case (ProvenFailedCQInfeas):
     logger_->MsgStream(LogInfo) << me_ << "constraint qualification "
                                 << "violated in node " << node->getId()
                                 << std::endl;
     ++stats_.prob;
   case (ProvenInfeasible):
   case (ProvenLocalInfeasible):
     node->setStatus(NodeInfeasible);
     should_prune = true;
     ++stats_.inf;
     break;

   case (ProvenObjectiveCutOff):
     node->setStatus(NodeHitUb);
     should_prune = true;
     ++stats_.ub;
     break;

   case (ProvenUnbounded):
     should_prune = false;
     logger_->MsgStream(LogDebug2) << me_ << "problem relaxation is "
                                   << "unbounded!" << std::endl;
     break;

   case (FailedFeas):
     logger_->MsgStream(LogInfo) << me_ << "Failed to converge " 
                                 << "(feasible) in node " << node->getId()
                                 << std::endl;
     if (node->getParent()) {
       node->setLb(node->getParent()->getLb());
     } else {
       node->setLb(-INFINITY);
     }
     node->setStatus(NodeContinue);
     ++stats_.prob;
     break;
   case (ProvenFailedCQFeas):
     logger_->MsgStream(LogInfo) << me_ << "constraint qualification "
                                 << "violated in node " << node->getId()
                                 << std::endl;
     if (node->getParent()) {
       node->setLb(node->getParent()->getLb());
     } else {
       node->setLb(-INFINITY);
     }
     node->setStatus(NodeContinue);
     ++stats_.prob;
     break;
   case (EngineIterationLimit):
     ++stats_.prob;
     logger_->MsgStream(LogInfo) << me_ << "engine hit iteration limit, "
                                 << "continuing in node " << node->getId()
                                 << std::endl;
     // continue with this node by following ProvenLocalOptimal case.
   case (ProvenLocalOptimal):
   case (ProvenOptimal):
     node->setLb(solval);
     if (solval >= s_pool->getBestSolutionValue() || solval >= cutOff_) {
       node->setStatus(NodeHitUb);
       should_prune = true;
       ++stats_.ub;
     } else {
       should_prune = false;
       node->setStatus(NodeContinue);
     }
     break;
   case (EngineError):
     if (contOnErr_) {
       logger_->MsgStream(LogError) << me_ << "engine reports error, "
                                    << " continuing in node " << node->getId()
                                    << std::endl;
       node->setStatus(NodeContinue);
       if (node->getParent()) {
         node->setLb(node->getParent()->getLb());
       } else {
         node->setLb(-INFINITY);
       }
     } else {
       logger_->MsgStream(LogError) << me_ << "engine reports error, "
                                    << "pruning node " << node->getId()
                                    << std::endl;
       should_prune = true;
       node->setStatus(NodeInfeasible);
       ++stats_.inf;
     }
     ++stats_.prob;
     break;
   default:
     break;
  }

  return should_prune;
}


void BndProcessor::solveRelaxation_() 
{
  engineStatus_ = EngineError;
  engine_->solve();
  engineStatus_ = engine_->getStatus();
#if SPEW
  logger_->MsgStream(LogDebug2) << me_ << "solving relaxation" << std::endl
                                << me_ << "engine status = " 
                                << engine_->getStatusString() << std::endl;
#endif
}


void BndProcessor::writeStats(std::ostream &out) const
{
  out << me_ << "nodes processed     = " << stats_.proc << std::endl 
      << me_ << "nodes branched      = " << stats_.bra << std::endl 
      << me_ << "nodes infeasible    = " << stats_.inf << std::endl 
      << me_ << "nodes optimal       = " << stats_.opt << std::endl 
      << me_ << "nodes hit ub        = " << stats_.ub << std::endl 
      << me_ << "nodes with problems = " << stats_.prob << std::endl 
      ;
}


void BndProcessor::writeStats() const
{
  writeStats(logger_->MsgStream(LogNone));
}


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
