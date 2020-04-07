// 
//     MINOTAUR -- It's only 1/2 bull
// 
//     (C)opyright 2010 - 2017 The MINOTAUR Team.
// 

/**
 * \file CxQuadHandler.cpp
 * \brief Implement the handler for functions of the general quadratic form 
 * \f$ \sum_i x^TAx \leq b \f$,
 * where \f$A\f$ may be indefinite.
 * \author Mustafa Vora, IIT Bombay
 */

#include <cmath>
#include <iostream>
#include <iomanip>

#include "MinotaurConfig.h"
#include "Branch.h"
#include "BrVarCand.h"
#include "Constraint.h"
#include "ConBoundMod.h"
#include "CxQuadHandler.h"
#include "Environment.h"
#include "Function.h"
#include "LinMods.h"
#include "Logger.h"
#include "Node.h"
#include "Objective.h"
#include "Operations.h"
#include "Option.h"
#include "QuadraticFunction.h"
#include "ProblemSize.h"
#include "Relaxation.h"
#include "SolutionPool.h"
#include "Timer.h"
#include "Variable.h"
#include <string.h>

//#define SPEW 1

using namespace Minotaur;

const std::string CxQuadHandler::me_ = "CxQuadHandler: ";

CxQuadHandler::CxQuadHandler(EnvPtr env, ProblemPtr problem, EnginePtr nlp_e)
  : aTol_(1e-5),
    bTol_(1e-8),
    eTol_(1e-8),
    rTol_(1e-4)
{
  env_ = env;
  vTol_ = env_->getOptions()->findDouble("feasAbs_tol")->getValue();
  problem_ = problem; 
  logger_  = env_->getLogger();
  nlpe_ = nlp_e;
  resetStats_();
  timer_ = env->getTimer(); 
}


CxQuadHandler::~CxQuadHandler()
{
  rel_ = 0;
  problem_ = 0;
  nlpe_ = 0;
  cvCons_.clear();
  cxCons_.clear();
}

void CxQuadHandler::addConstraint(ConstraintPtr newcon)
{
  Convexity cvx = newcon->getConvexity();
  LinearFunctionPtr lf = newcon->getFunction()->getLinearFunction();
  QuadraticFunctionPtr qf = newcon->getFunction()->getQuadraticFunction();
  VariablePtr y;
  LinQuadPtr lqf;
  
  if (cvx == Convex) {
    cxCons_.push_back(newcon);
    cons_.push_back(newcon);
  } else if (cvx == Concave) {
    assert(lf->getNumTerms() == 1);
    assert(lf->termsBegin()->second == -1.0);
    y = lf->termsBegin()->first;
    lqf = new LinQuad();
    lqf->qf = qf;
    lqf->y = y;
    lqf->sec = ConstraintPtr();
    cvCons_.push_back(lqf);
    cons_.push_back(newcon);
  } else if (cvx == Nonconvex) {
    assert(qf->getConvexity() != Nonconvex);
    assert(lf->getNumTerms() == 1);
    assert(lf->termsBegin()->second == -1.0);
    cxCons_.push_back(newcon);
    y = lf->termsBegin()->first;
    lqf = new LinQuad();
    lqf->qf = qf;
    lqf->y = y;
    lqf->sec = ConstraintPtr();
    cvCons_.push_back(lqf);
    cons_.push_back(newcon);
  } else {
    assert(!"Unknown Convexity constraint cannot be added to CxQuadHandler");
  }
}

SolveStatus CxQuadHandler::presolve(PreModQ *, bool *changed) {
  SolveStatus status = Started;
  ConstraintPtr c;
  QuadraticFunctionPtr qf;
  FunctionPtr f;
  VariablePtr y;
  Convexity cvx;
  double lb, ub;
  bool is_inf;
  double stime = timer_->query();

  *changed = false;

  for (LinQuadVecIter it = cvCons_.begin(); it != cvCons_.end();
       ++it) {
    y = (*it)->y;
    qf = (*it)->qf;
    propBounds_(qf, lb, ub); //returns false always
    cvx = qf->getConvexity();
    if (cvx == Convex) {
      /** lower bound of a convex quadratic function with no linear parts
       * cannot be less than 0
       */
      if (lb < bTol_) {
        lb = 0.0;
      }
    } else {
      /** upper bound of a concave quadratic function with no linear parts
       * cannot be more than 0
       */
      if (ub > bTol_) {
        ub = 0.0;
      }
    }
    if (updatePBounds_(y, lb, ub, changed) < 0) {
      status = SolvedInfeasible;
    }
  }

  for (ConstraintConstIterator it = cxCons_.begin(); it != cxCons_.end();
       ++it) {
    c = *it;
    f = c->getFunction();
    cvx = f->getQuadraticFunction()->getConvexity();
    propBounds_(f, lb, ub);
    if (lb > ub + vTol_ || ub < lb - vTol_ ||
        lb > c->getUb() + aTol_ || ub < c->getLb() - aTol_) {
      status = SolvedInfeasible;
    }
    if (cvx == Convex) {
      if (c->getUb() > ub + bTol_) {
        c->setUB(ub);
        *changed = true;
      }
    } else {
      if (c->getLb() < lb - bTol_) {
        c->setLB(lb);
        *changed = true;
      }
    }
  }
  
  if (status == Started) {
    status = Finished;
  }
  pStats_.time += timer_->query()-stime;
  return status;
}

bool CxQuadHandler::presolveNode(RelaxationPtr rel, NodePtr, SolutionPoolPtr,
		  ModVector &p_mods, ModVector &r_mods) {
  bool is_inf;
  VariablePtr y;
  ConstraintPtr c;
  FunctionPtr f;
  Convexity cvx;
  ConBoundModPtr cmod;
  QuadraticFunctionPtr qf;
  double lb, ub;
  double stime = timer_->query();
  bool changed = false;

  for (LinQuadVecIter it = cvCons_.begin(); it != cvCons_.end();
      ++it) {
    y = (*it)->y;
    qf = (*it)->qf;
    propBounds_(qf, lb, ub); //returns always false
    if (updatePBounds_(y, lb, ub, rel, &changed, true, p_mods, r_mods) < 0) {
      return true;
    }
  }

  for (ConstraintConstIterator it = cxCons_.begin(); it != cxCons_.end();
       ++it) {
    c = *it;
    f = c->getFunction();
    cvx = f->getQuadraticFunction()->getConvexity();
    propBounds_(f, lb, ub);
    if (lb > ub + vTol_ || ub < lb - vTol_ ||
        lb > c->getUb() + aTol_ || ub < c->getLb() - aTol_) {
      return true;
    }
    if (cvx == Convex) {
      if (c->getUb() > ub + bTol_) {
        cmod = (ConBoundModPtr) new ConBoundMod(c, Upper, ub);
        cmod->applyToProblem(problem_);
        p_mods.push_back(cmod);
        changed = true;
      }
    } else {
      if (c->getLb() < lb - bTol_) {
        cmod = (ConBoundModPtr) new ConBoundMod(c, Lower, lb);
        cmod->applyToProblem(problem_);
        p_mods.push_back(cmod);
        changed = true;
      }
    }
  }
}

void CxQuadHandler::relaxInitFull(RelaxationPtr rel, bool *is_inf)
{
  relax_(rel, is_inf);
}

void CxQuadHandler::relaxInitInc(RelaxationPtr rel, bool *is_inf)
{
  relax_(rel, is_inf);
}

void CxQuadHandler::relaxNodeFull(NodePtr, RelaxationPtr, bool *)
{
  assert(!"CxQuadHandler::relaxNodeFull not implemented!");
}

void CxQuadHandler::relaxNodeInc(NodePtr, RelaxationPtr, bool *)
{
  assert(!"CxQuadHandler::relaxNodeInc not implemented!");
}

bool CxQuadHandler::isFeasible(ConstSolutionPtr sol, RelaxationPtr, bool &,
                               double &) {
  int error = 0;
  double act;
  ConstraintPtr c;
  const double *x = sol->getPrimal();

  for (ConstraintConstIterator it = cons_.begin(); it != cons_.end();
      ++it) {
    c = *it;
    act = c->getActivity(x, &error);
    if (error == 0) {
      if (act > c->getUb() + eTol_ || act < c->getLb() - eTol_) {
        return false;
      }
    } else {
      logger_->msgStream(LogError) << me_ << c->getName() <<
        " constraint not defined at this point."<< std::endl;
      return false;
    }
  }
  return true;
}

void CxQuadHandler::separate(ConstSolutionPtr sol, NodePtr, RelaxationPtr,
                             CutManager *, SolutionPoolPtr, ModVector &,
                             ModVector &, bool *, SeparationStatus *status) {
  double act, ub, lb, c1;
  int error = 0;
  const double *x = sol->getPrimal();
  double *grad = 0;
  FunctionPtr f;
  ConstraintPtr c;
  Convexity cvx;
  std::stringstream sstm;
  LinearFunctionPtr lf = LinearFunctionPtr();
  double stime = timer_->query();

  grad = new double[rel_->getNumVars()];
  //y = new double[rel_->getNumVars()];
  for (ConstraintConstIterator it = cxCons_.begin(); it != cxCons_.end();
       ++it) {
    c = *it;
    act = c->getActivity(x, &error);
    if (error == 0) {
      f = c->getFunction();
      cvx = f->getQuadraticFunction()->getConvexity();
      memset(grad, 0, rel_->getNumVars()*sizeof(double));
      //memset(y, 0, rel_->getNumVars()*sizeof(double));
      f->evalGradient(x, grad, &error);
      ub = c->getUb();
      lb = c->getLb();
      if (cvx == Convex) {
        if (act > ub + eTol_) {
          ++(stats_->cuts);
          sstm << "_cxqCut_" << stats_->cuts;
          //findLinPt_(f, x, ub, grad, y, cvx);
          //act = c->getActivity(y, &error); 
          linearAt_(f, act, x, &c1, &lf, &error);
          f = (FunctionPtr) new Function();
          c = rel_->newConstraint(f, -INFINITY, ub-c1, sstm.str());
          *status = SepaResolve;
#if SPEW
          logger_->msgStream(LogDebug) << me_ << "added new cut" << std::endl;
          c->write(logger_->msgStream(LogDebug));
#endif
        }
      } else {
        if (act < lb - eTol_) {
          ++(stats_->cuts);
          sstm << "_cxqCut_" << stats_->cuts;
          //findLinPt_(f, x, lb, grad, y, cvx);
          //act = c->getActivity(y, &error);
          linearAt_(f, act, x, &c1, &lf, &error);
          f = (FunctionPtr) new Function();
          c = rel_->newConstraint(f, lb-c1, INFINITY, sstm.str());
          *status = SepaResolve;
#if SPEW
          logger_->msgStream(LogDebug) << me_ << "added new cut" << std::endl;
          c->write(logger_->msgStream(LogDebug));
#endif
        }
      }
    }
  }
  sStats_.time += timer_->query()-stime;
}

ModificationPtr CxQuadHandler::getBrMod(BrCandPtr cand, DoubleVector &x,
                                        RelaxationPtr rel, BranchDirection dir)
{
  LinModsPtr lmods = (LinModsPtr) new LinMods();
  LinConModPtr lmod;
  BrVarCandPtr vcand = dynamic_cast <BrVarCand*> (cand);
  VariablePtr v = vcand->getVar();
  UInt vind = v->getIndex();
  ConstraintPtr c;
  double *z;
  QuadraticFunctionPtr qf;
  double ext_qf, lb, ub;
  std::vector<BoundType> ext_b;
  DoubleVector evals;

  if (dir == DownBranch) {
    lb = v->getLb();
    ub = x[vind];
  } else {
    lb = x[vind];
    ub = v->getUb();
  }

  for (LinQuadVecIter it = cvCons_.begin(); it != cvCons_.end();
       ++it) {
    qf = (*it)->qf;
    z = new double[rel->getNumVars()];
    memset(z, 0, rel->getNumVars()*sizeof(double));
    for (VarIntMapConstIterator qit = qf->varsBegin();
         qit != qf->varsEnd(); ++qit) {
      if (qit->first->getIndex() == vind) {
        findExtPt_((*it), &ext_qf, ext_b, z, evals, lb, ub, v);
        addSecant_((*it), ext_qf, ext_b, evals, rel, lb, ub, v, lmod);
        if (lmod) {
          lmods->insert(lmod);
        }
        break;
      }
    }
  }
  return lmods;
}

void CxQuadHandler::addInitLinearX_(const double *x) {
  int error=0;
  FunctionPtr f;
  double c, act;
  std::stringstream sstm;
  ConstraintPtr con;
  //ConstraintPtr newcon;
  LinearFunctionPtr lf = LinearFunctionPtr();

  for (ConstraintConstIterator it = cxCons_.begin();
      it != cxCons_.end(); ++it) {
    con = *it;
    act = con->getActivity(x, &error);
    
    if (error == 0) {
      f = con->getFunction();
      linearAt_(f, act, x, &c, &lf, &error);
      if (error == 0) {
        ++(stats_->cuts);
        sstm << "_cxqCutRoot_" << stats_->cuts;
        if (f->getQuadraticFunction()->getConvexity() == Convex) {
          f = (FunctionPtr) new Function(lf);
          con = rel_->newConstraint(f, -INFINITY, con->getUb()-c, sstm.str());
        } else {
          f = (FunctionPtr) new Function(lf);
          con = rel_->newConstraint(f, con->getLb()-c, INFINITY, sstm.str());
        }
#if SPEW
        logger_->msgStream(LogDebug) << me_ << "added new cut" << std::endl;
        con->write(logger_->msgStream(LogDebug));
#endif
        //newcon = rel_->newConstraint(f, -INFINITY, cUb-c, sstm.str());
        sstm.str("");
      }
    }	else {
      logger_->msgStream(LogError) << me_ << "Constraint" <<  con->getName() <<
        " is not defined at this point." << std::endl;
    }
  }
}

void CxQuadHandler::addSecant_(LinQuadPtr lqf, double rhs,
                               std::vector<BoundType> b, DoubleVector evals) {
  std::vector<BoundType> b1;
  double alpha, alpha_ext;
  double x_val = 0.0, b_val = 0.0;
  QuadraticFunctionPtr qf = lqf->qf;
  VariablePtr y = rel_->getRelaxationVar(lqf->y);
  Convexity cvx = qf->getConvexity();
  UInt i, j, numvars = qf->getNumVars();
  VarIntMapConstIterator it;
  LinearFunctionPtr lf = (LinearFunctionPtr) new LinearFunction();
  FunctionPtr fnew;
  ConstraintPtr cnew;
  
  lf->addTerm(y, 1.0);

  i = 0;
  for (it = qf->varsBegin(); it != qf->varsEnd(); ++it) {
    if (b[i] == Lower) {
      b_val += it->first->getLb();
      x_val += it->first->getLb();
    } else {
      b_val -= it->first->getUb();
      x_val -= it->first->getLb();
    }
    b1.push_back(Lower);
    ++i;
  }

  i = 0;
  alpha_ext = fabs(rhs - evals[i]) > eTol_ ?
              (rhs - evals[i])/(x_val - b_val) : 0.0;
  ++i;

  while (true) {
    it = qf->varsBegin();
    j = 0;
    while (b1[j] == Upper) {
      b1[j] = Lower;
      x_val += (b[j] == Lower) ?
               it->first->getLb() - it->first->getUb() :
               it->first->getUb() - it->first->getLb();
      ++it;
      ++j;
    }
    if (j < numvars) {
      b1[j] = Upper;
      x_val -= (b[j] == Lower) ?
               it->first->getLb() - it->first->getUb() :
               it->first->getUb() - it->first->getLb();
      if (fabs(rhs - evals[i]) > eTol_) {
        alpha = (rhs - evals[i])/(x_val - b_val);
      } else {
        if (b1 == b) {
          ++i;
          continue;
        }
        else {
          alpha = 0.0;
        }
      }
      if (cvx == Convex && alpha < alpha_ext) {
        alpha_ext = alpha;
      }
      if (cvx == Concave && alpha > alpha_ext) {
        alpha_ext = alpha;
      }
    } else {
      break;
    }
    ++i;
  }
  for (it = qf->varsBegin(); it != qf->varsEnd(); ++it) {
    lf->addTerm(rel_->getRelaxationVar(it->first), alpha_ext);
  }
  fnew = (FunctionPtr) new Function(lf);
  if (cvx == Convex) {
    cnew = rel_->newConstraint(fnew, -INFINITY, rhs+alpha_ext*b_val);
    lqf->sec = cnew;
  } else {
    cnew = rel_->newConstraint(fnew, rhs+alpha_ext*b_val, INFINITY);
    lqf->sec = cnew;
  }
  b1.clear();
}

void CxQuadHandler::addSecant_(LinQuadPtr lqf, double rhs,
                               std::vector<BoundType> b, DoubleVector evals,
                               RelaxationPtr rel, double vlb, double vub,
                               VariablePtr v, LinConModPtr lmod) {
  std::vector<BoundType> b1;
  double alpha, alpha_ext;
  double x_val = 0.0, b_val = 0.0;
  QuadraticFunctionPtr qf = lqf->qf;
  VariablePtr y = rel->getRelaxationVar(lqf->y);
  Convexity cvx = qf->getConvexity();
  UInt i, j, numvars = qf->getNumVars();
  VarIntMapConstIterator it;
  LinearFunctionPtr lf = (LinearFunctionPtr) new LinearFunction();
  
  lf->addTerm(y, 1.0);

  i = 0;
  for (it = qf->varsBegin(); it != qf->varsEnd(); ++it) {
    if (b[i] == Lower) {
      b_val += it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
      x_val += it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
    } else {
      b_val -= it->first->getIndex() == v->getIndex() ? vub : it->first->getUb();
      x_val -= it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
    }
    b1.push_back(Lower);
    ++i;
  }

  i = 0;
  alpha_ext = fabs(rhs - evals[i]) > eTol_ ?
              (rhs - evals[i])/(x_val - b_val) : 0.0;
  ++i;

  while (true) {
    it = qf->varsBegin();
    j = 0;
    while (b1[j] == Upper) {
      b1[j] = Lower;
      x_val += (b[j] == Lower) ?
               it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb()
             - it->first->getIndex() == v->getIndex() ? vub : it->first->getUb() :
               it->first->getIndex() == v->getIndex() ? vub : it->first->getUb()
             - it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
      ++it;
      ++j;
    }
    if (j < numvars) {
      b1[j] = Upper;
      x_val -= (b[j] == Lower) ?
               it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb()
             - it->first->getIndex() == v->getIndex() ? vub : it->first->getUb() :
               it->first->getIndex() == v->getIndex() ? vub : it->first->getUb()
             - it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
      if (fabs(rhs - evals[i]) > eTol_) {
        alpha = (rhs - evals[i])/(x_val - b_val);
      } else {
        if (b1 == b) {
          ++i;
          continue;
        }
        else {
          alpha = 0.0;
        }
      }
      if (cvx == Convex && alpha < alpha_ext) {
        alpha_ext = alpha;
      }
      if (cvx == Concave && alpha > alpha_ext) {
        alpha_ext = alpha;
      }
    } else {
      break;
    }
    ++i;
  }
  for (it = qf->varsBegin(); it != qf->varsEnd(); ++it) {
    lf->addTerm(rel->getRelaxationVar(it->first), alpha_ext);
  }
  if (cvx == Convex) {
    lmod = (LinConModPtr) new LinConMod(lqf->sec, lf, -INFINITY,
                                        rhs+alpha_ext*b_val);
  } else {
    lmod = (LinConModPtr) new LinConMod(lqf->sec, lf, rhs+alpha_ext*b_val,
                                        INFINITY);
  }
  b1.clear();
}

void CxQuadHandler::createConvexRelaxation_() {
  ConstraintPtr c, cnew;
  QuadraticFunctionPtr qf;
  FunctionPtr f, fnew;
  int error = 0;
  
  for (ConstraintConstIterator cit = cxCons_.begin();
      cit != cxCons_.end(); ++cit) {
    c = *cit;
    f = c->getFunction();
    qf = f->getQuadraticFunction();
    fnew = f->cloneWithVars(rel_->varsBegin(), &error);
    if (qf->getConvexity() == Convex) {
      cnew = rel_->newConstraint(fnew, -INFINITY, c->getUb());
    } else {
      cnew = rel_->newConstraint(fnew, c->getLb(), INFINITY);
    }
  }
}

void CxQuadHandler::deleteQuadConsfromRel_() {
  ConstraintPtr c;
  FunctionType ftype;
  
  for (ConstraintConstIterator cit = rel_->consBegin();
       cit != rel_->consEnd(); ++cit) {
    c = *cit;
    ftype = c->getFunction()->getType();
    if (ftype != Linear && ftype != Constant) {
      assert(ftype == Quadratic || ftype == Bilinear);
      rel_->markDelete(c);
    }
  }
  rel_->delMarkedCons();
}

void CxQuadHandler::findExtPt_(LinQuadPtr lqf, double *ext_qf,
                               std::vector<BoundType> &ext_b,
                               double *x, DoubleVector &evals) {
  std::vector<BoundType> b;
  double lb, ub;
  QuadraticFunctionPtr qf = lqf->qf;
  Convexity cvx = qf->getConvexity();
  UInt i, numvars = qf->getNumVars();
  double eval_qf;

  for (VarIntMapConstIterator it = qf->varsBegin();
      it != qf->varsEnd(); ++it) {
    lb = it->first->getLb();
    ub = it->first->getUb();
    if (lb <= -INFINITY || ub >= INFINITY) {
      logger_->errStream() << "can not relax "
                           << it->first->getName()
                           << " because bounds on the variable is too weak"
                           << std::endl;
      exit(500);
    }
    x[it->first->getIndex()] = lb;
    b.push_back(Lower);
    ext_b.push_back(Lower);
  }
  *ext_qf = qf->eval(x);
  evals.push_back(*ext_qf);
  while (true) {
    if (cvx == Convex) {
      i = 0;
      VarIntMapConstIterator it = qf->varsBegin();
      while (b[i] == Upper) {
        b[i] = Lower;
        x[it->first->getIndex()] = it->first->getLb();
        ++it;
        ++i;
      }
      if (i < numvars) {
        b[i] = Upper;
        x[it->first->getIndex()] = it->first->getUb();
        eval_qf = qf->eval(x);
        evals.push_back(eval_qf);
        if (eval_qf > *ext_qf) {
          ext_b = b;
          *ext_qf = eval_qf;
        }
      } else {
        break;
      }
    } else {
      i = 0;
      VarIntMapConstIterator it = qf->varsBegin();
      while (b[i] == Upper) {
        b[i] = Lower;
        x[it->first->getIndex()] = it->first->getLb();
        ++it;
        ++i;
      }
      if (i < numvars) {
        b[i] = Upper;
        x[it->first->getIndex()] = it->first->getUb();
        eval_qf = qf->eval(x);
        evals.push_back(eval_qf);
        if (eval_qf < *ext_qf) {
          ext_b = b;
          *ext_qf = eval_qf;
        }
      } else {
        break;
      }
    }
  }
  b.clear();
}

void CxQuadHandler::findExtPt_(LinQuadPtr lqf, double *ext_qf,
                               std::vector<BoundType> &ext_b, double *x,
                               DoubleVector &evals, double vlb,
                               double vub, VariablePtr v) {
  std::vector<BoundType> b;
  double lb, ub;
  QuadraticFunctionPtr qf = lqf->qf;
  Convexity cvx = qf->getConvexity();
  UInt i, numvars = qf->getNumVars();
  double eval_qf;

  for (VarIntMapConstIterator it = qf->varsBegin();
      it != qf->varsEnd(); ++it) {
    lb = it->first->getIndex() == v->getIndex() ? vlb : it->first->getLb();
    ub = it->first->getIndex() == v->getIndex() ? vub : it->first->getUb();
    x[it->first->getIndex()] = lb;
    b.push_back(Lower);
    ext_b.push_back(Lower);
  }
  *ext_qf = qf->eval(x);
  evals.push_back(*ext_qf);
  while (true) {
    if (cvx == Convex) {
      i = 0;
      VarIntMapConstIterator it = qf->varsBegin();
      while (b[i] == Upper) {
        b[i] = Lower;
        x[it->first->getIndex()] = it->first->getLb();
        ++it;
        ++i;
      }
      if (i < numvars) {
        b[i] = Upper;
        x[it->first->getIndex()] = it->first->getUb();
        eval_qf = qf->eval(x);
        evals.push_back(eval_qf);
        if (eval_qf > *ext_qf) {
          ext_b = b;
          *ext_qf = eval_qf;
        }
      } else {
        break;
      }
    } else {
      i = 0;
      VarIntMapConstIterator it = qf->varsBegin();
      while (b[i] == Upper) {
        b[i] = Lower;
        x[it->first->getIndex()] = it->first->getLb();
        ++it;
        ++i;
      }
      if (i < numvars) {
        b[i] = Upper;
        x[it->first->getIndex()] = it->first->getUb();
        eval_qf = qf->eval(x);
        evals.push_back(eval_qf);
        if (eval_qf < *ext_qf) {
          ext_b = b;
          *ext_qf = eval_qf;
        }
      } else {
        break;
      }
    }
  }
  b.clear();
}

void CxQuadHandler::initLinear_(bool *isInf) {
  *isInf = false;
  const double *x;
  
  rel_->setNativeDer();
  rel_->prepareForSolve();
  nlpe_->load(rel_);
  solveQCQP_();
  
  switch (nlpStatus_) {
  case (ProvenOptimal):
  case (ProvenLocalOptimal):
    ++(stats_->nlpF);
    x = nlpe_->getSolution()->getPrimal();
    addInitLinearX_(x);
    break;
  case (EngineIterationLimit):
    ++(stats_->nlpIL);
    x = nlpe_->getSolution()->getPrimal();
    addInitLinearX_(x);
    break;
  case (ProvenInfeasible):
  case (ProvenLocalInfeasible):
  case (ProvenObjectiveCutOff):
    ++(stats_->nlpI);
    *isInf = true;
    break;
  case (FailedFeas):
  case (EngineError):
  case (FailedInfeas):
  case (ProvenUnbounded):
  case (ProvenFailedCQFeas):
  case (EngineUnknownStatus):
  case (ProvenFailedCQInfeas):
  default:
    logger_->msgStream(LogError) << me_ << "NLP engine status at root= "
      << nlpStatus_ << std::endl;
    assert(!"In CxQuadHandler: stopped at root. Check error log.");
    break;
  }
  return;
}

void CxQuadHandler::linearAt_(FunctionPtr f, double fval, const double *x,
                          double *c, LinearFunctionPtr *lf, int *error)
{
  int n = rel_->getNumVars();
  double *a = new double[n];
  VariableConstIterator vbeg = rel_->varsBegin(), vend = rel_->varsEnd();

  std::fill(a, a+n, 0.);
  f->evalGradient(x, a, error);
  
  if (*error==0) {
    *lf = (LinearFunctionPtr) new LinearFunction(a, vbeg, vend, vTol_);
    *c  = fval - InnerProduct(x, a, n);
  } else {
    logger_->msgStream(LogError) << me_ <<"gradient not defined at this point."
      << std::endl;
  }
  delete [] a;
  return;
}

void CxQuadHandler::propBounds_(QuadraticFunctionPtr qf, double &lb,
                                double &ub) {
  VariablePtr x0,x1;
  double l, u, coeff, swap;

  lb = 0.0;
  ub = 0.0;
  for (VariablePairGroupConstIterator it = qf->begin(); it != qf->end();
       ++it) {
    x0 = it->first.first;
    x1 = it->first.second;
    coeff = it->second;
    if (x0->getIndex() == x1->getIndex()) {
      BoundsOnSquare(x0, l, u);
    } else {
      BoundsOnProduct(true, x0, x1, l, u);
    }
    if (coeff < 0) {
      swap = l;
      l = coeff*u;
      u = coeff*swap;
    } else {
      l = coeff*l;
      u = coeff*u;
    }
    lb += l;
    ub += u;
  }
}

void CxQuadHandler::propBounds_(LinearFunctionPtr lf, double &lb, double &ub) {
  double l, u, coeff, swap;

  lb = 0.0;
  ub = 0.0;

  for (VariableGroupConstIterator it = lf->termsBegin(); it != lf->termsEnd();
       ++it) {
    l = it->first->getLb();
    u = it->first->getUb();
    coeff = it->second;

    if (coeff < 0) {
      swap = l;
      l = coeff*u;
      u = coeff*swap;
    } else {
      l = coeff*l;
      u = coeff*u;
    }
    lb += l;
    ub += u;
  }
}

void CxQuadHandler::propBounds_(FunctionPtr f, double &lb, double &ub) {
  double l, u;

  lb = 0.0;
  ub = 0.0;
  
  propBounds_(f->getQuadraticFunction(), l, u);
  lb += l;
  ub += u;

  propBounds_(f->getLinearFunction(), l, u);
  lb += l;
  ub += u;
}

void CxQuadHandler::relax_(RelaxationPtr rel, bool *isInf) {
  rel_ = rel;
  relaxConcaveCons_();
  if (!cxCons_.empty()) {
    createConvexRelaxation_();
    initLinear_(isInf);
    deleteQuadConsfromRel_();
    rel_->unsetEngine();
  }
}

void CxQuadHandler::relaxConcaveCons_() {
  double *x;
  std::vector<BoundType> ext_b;
  double ext_qf;
  DoubleVector evals;

  x = new double[rel_->getNumVars()];
  memset(x, 0, rel_->getNumVars()*sizeof(double));

  for (LinQuadVecIter it = cvCons_.begin();
       it != cvCons_.end(); ++it) {
    findExtPt_((*it), &ext_qf, ext_b, x, evals);
    addSecant_((*it), ext_qf, ext_b, evals);
  }
  delete [] x;
  ext_b.clear();
  evals.clear();
}

void CxQuadHandler::resetStats_()
{
  pStats_.iters  = 0;
  pStats_.time   = 0.0;
  pStats_.timeN  = 0.0;
  pStats_.vBnd   = 0;
  pStats_.nMods  = 0;
  
  stats_ = new CxQStats();
  stats_->cuts = 0;
  stats_->nlpS = 0;
  stats_->nlpF = 0;
  stats_->nlpI = 0;
  stats_->nlpIL = 0;
}

void CxQuadHandler::solveQCQP_()
{
  nlpStatus_ = nlpe_->solve();
  ++(stats_->nlpS);
  return;
}

int CxQuadHandler::updatePBounds_(VariablePtr v, double lb, double ub,
                                bool *changed)
{
  if (ub < v->getLb() - bTol_ || lb > v->getUb() + bTol_) { 
#if SPEW
      logger_->msgStream(LogDebug2) << me_ << "inconsistent bounds of"
                                    << v->getName() << " " << v->getLb()
                                    << " " << v->getUb() << std::endl;
#endif
    return -1;
  }
  
  if (ub < v->getUb() - bTol_ && (v->getUb() == INFINITY ||
                                  ub < v->getUb()-fabs(v->getUb())*rTol_)) {
    problem_->changeBound(v, Upper, ub);
    *changed = true;
    ++pStats_.vBnd;
#if SPEW
      logger_->msgStream(LogDebug2) << me_ << "new ub of " << v->getName()
                                    << " = " << v->getUb() << std::endl;
#endif
  }
  if (lb > v->getLb() + aTol_ && (v->getLb() == -INFINITY ||
                                  lb > v->getLb()+fabs(v->getLb())*rTol_)) {
    problem_->changeBound(v, Lower, lb);
    *changed = true;
    ++pStats_.vBnd;
#if SPEW
      logger_->msgStream(LogDebug2) << me_ << "new lb of " << v->getName()
                                    << " = " << v->getLb() << std::endl;
#endif
  }

  return 0;
}

int CxQuadHandler::updatePBounds_(VariablePtr v, double lb, double ub,
                                  RelaxationPtr rel, bool *changed,
                                  bool mod_rel, ModVector &p_mods,
                                  ModVector &r_mods) {
  VarBoundMod2Ptr b2mod;
  VarBoundModPtr bmod;

  if (ub < v->getLb() - bTol_ || lb > v->getUb() + bTol_) { 
    return -1;
  }
  
  if (ub < v->getUb() - bTol_ && (v->getUb() == INFINITY ||
                                  ub < v->getUb()-fabs(v->getUb())*rTol_) &&
      lb > v->getLb() + aTol_ && (v->getLb() == -INFINITY ||
                                  lb > v->getLb()+fabs(v->getLb())*rTol_)) {
    *changed = true;
    ++pStats_.vBnd;
    b2mod = (VarBoundMod2Ptr) new VarBoundMod2(v, lb, ub);
    b2mod->applyToProblem(problem_);
    p_mods.push_back(b2mod);
#if SPEW
    b2mod->write(logger_->msgStream(LogDebug2));
#endif
    if (mod_rel) {
      b2mod = (VarBoundMod2Ptr)
              new VarBoundMod2(rel->getRelaxationVar(v), lb, ub);
      b2mod->applyToProblem(rel);
      r_mods.push_back(b2mod);
    }
  } else if (lb > v->getLb()+bTol_ && 
             (v->getLb()==-INFINITY || lb>v->getLb()+rTol_*fabs(v->getLb()))) {
    ++pStats_.vBnd;
    *changed = true;
    bmod  = (VarBoundModPtr) new VarBoundMod(v, Lower, lb);
    bmod->applyToProblem(problem_);
    p_mods.push_back(bmod);
#if SPEW
    bmod->write(logger_->msgStream(LogDebug2));
#endif 

    if (true == mod_rel) {
      bmod = (VarBoundModPtr)
             new VarBoundMod(rel->getRelaxationVar(v), Lower, lb);
      bmod->applyToProblem(rel);
      r_mods.push_back(bmod);
    }
  } else if (ub < v->getUb()-bTol_ &&
             (v->getUb()== INFINITY || ub<v->getUb()-rTol_*fabs(v->getUb()))) {
    ++pStats_.vBnd;
    *changed = true;
    bmod  = (VarBoundModPtr) new VarBoundMod(v, Upper, ub);
    bmod->applyToProblem(problem_);
    p_mods.push_back(bmod);
#if SPEW
    bmod->write(logger_->msgStream(LogDebug2));
#endif 
    
    if (true == mod_rel) {
      bmod  = (VarBoundModPtr)
               new VarBoundMod(rel->getRelaxationVar(v), Upper, ub);
      bmod->applyToProblem(rel);
      r_mods.push_back(bmod);
    }
  } 
  return 0;
}

void CxQuadHandler::writeStats(std::ostream &out) const
{
  out
    << me_ << "number of nlps solved                       = "
    << stats_->nlpS << std::endl
    << me_ << "number of infeasible nlps                   = " 
    << stats_->nlpI << std::endl
    << me_ << "number of feasible nlps                     = " 
    << stats_->nlpF << std::endl
    << me_ << "number of nlps hit engine iterations limit  = " 
    << stats_->nlpIL << std::endl
    << me_ << "number of cuts added                        = " 
    << stats_->cuts << std::endl;
  return;
}

std::string CxQuadHandler::getName() const
{
   return "CxQuadHandler (Handling convexifiable quadratic constraints).";
}
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
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
