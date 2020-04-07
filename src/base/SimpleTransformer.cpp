//
//     MINOTAUR -- It's only 1/2 bull
//
//     (C)opyright 2008 - 2017 The MINOTAUR Team.
//

/**
 * \file SimpleTransformer.cpp
 * \brief Define class for simple reformulations a problem to make it suitable
 * for handlers.
 * \author Ashutosh Mahajan, Argonne National Laboratory
 */

#include <cmath>
#include <iostream>

#include "MinotaurConfig.h"

#include "Environment.h"
#include "CGraph.h"
#include "CNode.h"
#include "Constraint.h"
#include "CxUnivarHandler.h"
#include "CxQuadHandler.h"
#include "Function.h"
#include "IntVarHandler.h"
#include "LinBil.h"
#include "LinearFunction.h"
#include "LinearHandler.h"
#include "Logger.h"
#include "NonlinearFunction.h"
#include "Option.h"
#include "Objective.h"
#include "Problem.h"
#include "ProblemSize.h"
#include "QuadraticFunction.h"
#include "QuadHandler.h"
#include "SimpleTransformer.h"
#include "Solution.h"
#include "Timer.h"
#include "Variable.h"
#include "YEqCGs.h"
#include "YEqLFs.h"
#include "YEqUCGs.h"
#include "YEqVars.h"
#include "YEqQf.h"

// #define SPEW 1

using namespace Minotaur;
const std::string SimpleTransformer::me_ = "SimpleTransformer: ";


SimpleTransformer::SimpleTransformer()
  : Transformer(),
    yBiVars_(0),
    yQf_(0),
    nlpe_(0)
{
}


SimpleTransformer::SimpleTransformer(EnvPtr env, ConstProblemPtr p, EnginePtr nlp_e)
  : Transformer(env, p),
    yBiVars_(0),
    yQf_(0)
{
  nlpe_ = nlp_e;
}


SimpleTransformer::~SimpleTransformer() 
{
  delete yBiVars_;
  delete yQf_;
  nlpe_ = 0;
}


void SimpleTransformer::absRef_(LinearFunctionPtr lfl, VariablePtr vl,
                                double dl, VariablePtr &v, double &d)
{
  if (lfl) {
    vl = newVar_(lfl, dl, newp_);
  } else if (vl && fabs(dl)>zTol_) {
    vl = newVar_(vl, dl, newp_);
  }
  if (vl) {
    CGraphPtr cg = (CGraphPtr) new CGraph();
    CNode *n1 = cg->newNode(vl);
    CNode *n2 = 0;

    n1 = cg->newNode(OpAbs, n1, n2);
    cg->setOut(n1);
    cg->finalize();
    v = newVar_(cg, newp_);
  } else {
    d = fabs(dl);
  }
}


void SimpleTransformer::bilRef_(LinearFunctionPtr lfl, VariablePtr vl,
                                double dl, LinearFunctionPtr lfr,
                                VariablePtr vr, double dr,
                                LinearFunctionPtr &lf, VariablePtr &v,
                                double &d)
{
  if (lfl) {
    vl = newVar_(lfl, dl, newp_);
    if (vr) {
      vr = newVar_(vr, dr, newp_);
    } else if (lfr) {
      vr = newVar_(lfr, dr, newp_);
    } 
    if (vr) {
      //lf.reset();
      lf = 0;
      d = 0;
      v = newBilVar_(vl, vr);
    } else {
      lf = lfl;
      lf->multiply(dr);
      d = dl*dr;
      v = 0;
    }
  } else if (vl) {
    vl = newVar_(vl, dl, newp_);
    if (lfr) {
      vr = newVar_(lfr, dr, newp_);
    } else if (vr) {
      vr = newVar_(vr, dr, newp_);
    } 
    if (vr) {
      v = newBilVar_(vl, vr);
      //lf.reset();
      lf = 0;
      d = 0;
    } else {
      lf = (LinearFunctionPtr) new LinearFunction();
      lf->addTerm(vl, dr);
      v = 0;
      d = 0;
    }
  } else if (lfr) {
    lf = lfr;
    lf->multiply(dl);
    d = dl*dr;
    v = 0;
  } else if (vr) {
    lf = (LinearFunctionPtr) new LinearFunction();
    lf->addTerm(vr, dl);
    v = 0;
    d = 0;
  } else {
    //lf.reset();
    lf = 0;
    v = 0;
    d = dl*dr;
  }
}


std::string SimpleTransformer::getName() const
{
  return "SimpleTransformer";
}


SolutionPtr SimpleTransformer::getSolOrig(ConstSolutionPtr, int &err )
{
  err = 1;
  return SolutionPtr();
}


SolutionPtr SimpleTransformer::getSolTrans(ConstSolutionPtr, int &err )
{
  err = 1;
  return SolutionPtr();
}


VariablePtr SimpleTransformer::newBilVar_(VariablePtr vl, VariablePtr vr)
{
  CGraphPtr cg = (CGraphPtr) new CGraph();
  CNode *n1 = cg->newNode(vl);
  CNode *n2 = 0;
  VariablePtr ov = VariablePtr();
  LinearFunctionPtr lf;
  FunctionPtr f;
  ConstraintPtr cnew;

  if (vl == vr) {
    n2 = cg->newNode(OpSqr, n1, n2);
  } else {
    n2 = cg->newNode(vr);
    n2 = cg->newNode(OpMult, n1, n2);
  }
  cg->setOut(n2);
  cg->finalize();

  if (vl == vr) {
    ov = newVar_(cg, newp_);
  } else {
    ov = yBiVars_->findY(cg);
    if (!ov) {
      ov = newp_->newVariable();
      lf = (LinearFunctionPtr) new LinearFunction();
      lf->addTerm(ov, -1.0);
      f = (FunctionPtr) new Function(lf, cg);
      cnew = newp_->newConstraint(f, 0.0, 0.0);
#if SPEW
      logger_->msgStream(LogDebug) << me_ << "added new constraint"
                                   << std::endl;
      cnew->write(logger_->msgStream(LogDebug));
#endif 
      qHandler_->addConstraint(cnew);
      yBiVars_->insert(ov, cg);
    }
  }
  return ov;
}


void SimpleTransformer::powKRef_(LinearFunctionPtr lfl,
                                 VariablePtr vl, double dl, double k,
                                 LinearFunctionPtr &lf, VariablePtr &v,
                                 double &d)
{
  CNode *n1, *n2;
  if (fabs(k-floor(k+0.5))>zTol_) {
   assert(!"fractional powers can not be handled yet!");
  } else if (k<-zTol_) {
   assert(!"negative powers can not be handled yet!");
  } else if (fabs(k/2 - floor(k/2+0.5))>zTol_) {
   logger_->errStream() << "odd powers can not be handled yet!" << std::endl;
  }

  if (lfl) {
    vl = newVar_(lfl, dl, newp_);
  }
  if (vl) {
    CGraphPtr cg = (CGraphPtr) new CGraph();
    n1 = cg->newNode(vl);
    n2 = cg->newNode(k);
    n2 = cg->newNode(OpPowK, n1, n2);
    cg->setOut(n2);
    cg->finalize();
    v = 0;
    //lf.reset();
    lf = 0;
    v = newVar_(cg, newp_);
    d = 0;
  } else {
    d = pow(dl, k);
  }
}


// Returns one of the following four:
// #1 lf + d, 
// #2  v + d, or
// d.
// d may be zero, lf and v may simultaneously be NULL. 
// TODO: return an error code if there is an error?
void SimpleTransformer::recursRef_(const CNode *node, LinearFunctionPtr &lf,
                                   VariablePtr &v, double &d)
{
  double dl = 0;
  double dr = 0;
  LinearFunctionPtr lfl = LinearFunctionPtr();
  LinearFunctionPtr lfr = LinearFunctionPtr();
  VariablePtr vl = VariablePtr();
  VariablePtr vr = VariablePtr();
  VariablePtr v2 = VariablePtr();
  CNode *n1 = 0;

  lf = LinearFunctionPtr(); // NULL
  v = VariablePtr();
  d = 0.0;

  switch (node->getOp()) {
  case (OpAbs):
  case (OpAcos):
  case (OpAcosh):
  case (OpAsin):
  case (OpAsinh):
  case (OpAtan):
  case (OpAtanh):
  case (OpCeil):
  case (OpCos):
  case (OpCosh):
  case (OpCPow):
    recursRef_(node->getL(), lfl, vl, dl);
    uniVarRef_(node, lfl, vl, dl, lf, v, d);
    break;
  case (OpDiv):
    // (lfl+vl+dl)/(lfr+vr+dr), there are many sub-cases
    recursRef_(node->getL(), lfl, vl, dl);
    recursRef_(node->getR(), lfr, vr, dr);

    if (!lfl && !vl && !lfr && !vr && fabs(dl)<zTol_ && fabs(dr)<zTol_) {
      logger_->msgStream(LogDebug) << "seeing zero by zero" << std::endl;
    } else if (!lfr && !vr && fabs(dr)<zTol_) {
      logger_->msgStream(LogDebug) << "seeing division by zero" << std::endl;
    } else if (!lfl && !vl && fabs(dl)<zTol_) {
      d = 0.0;
    } else if (!lfr && !vr && fabs(dr-1.0)<zTol_) {
      d = dl;
      lf = lfl;
      v = vl;
    } else if (!lfr && !vr) {
      d = dl/dr;
      lf = lfr->clone(); 
      lf->multiply(1.0/dr);
    } else {
      CGraphPtr cg = (CGraphPtr) new CGraph();
      CNode *n2 = 0;
      if (lfr) {
        v2 = newVar_(lfr, dr, newp_);
      } else {
        v2 = newVar_(vr, dr, newp_);
      }

      // 1/v2
      n1 = cg->newNode(1.0);
      n2 = cg->newNode(v2);
      n1 = cg->newNode(OpDiv, n1, n2);
      cg->setOut(n1);
      cg->finalize();
      v2 = newVar_(cg, newp_);

      //lfr.reset();
      lfr = 0;
      // now we have to do (lfl + vl + dl)*v2
      bilRef_(lfl, vl, dl, lfr, v2, 0.0, lf, v, d);
    }
    break;
  case (OpExp):
  case (OpFloor):
  case (OpInt):
  case (OpIntDiv):
  case (OpLog):
  case (OpLog10):
    recursRef_(node->getL(), lfl, vl, dl);
    uniVarRef_(node, lfl, vl, dl, lf, v, d);
    break;
  case (OpMinus):
    recursRef_(node->getL(), lfl, vl, dl);
    recursRef_(node->getR(), lfr, vr, dr);
    d = dl - dr;
    if (!vr && !lfr) {
      v = vl;
      lf = lfl;
    } else if (!vl && !lfl) {
      if (lfr) {
        lf = lfr;
        lf->multiply(-1.0);
      } else if (vr) {
        lf = (LinearFunctionPtr) new LinearFunction();
        lf->addTerm(vr, -1.0);
      }
    } else {
      lf = (LinearFunctionPtr) new LinearFunction();
      if (lfl) {
        lf->add(lfl);
      } else if (vl) {
        lf->incTerm(vl, 1.0);
      }
      if (lfr) {
        lfr->multiply(-1.0);
        lf->add(lfr);
      } else if (vr) {
        lf->incTerm(vr, -1.0);
      }
      v = 0;
    }
    break;
  case (OpMult):
    recursRef_(node->getL(), lfl, vl, dl);
    recursRef_(node->getR(), lfr, vr, dr);
    bilRef_(lfl, vl, dl, lfr, vr, dr, lf, v, d);
    break;
  case (OpNone):
    break;
  case (OpNum):
    d = node->getVal();
    break;
  case (OpPlus):
    recursRef_(node->getL(), lfl, vl, dl);
    recursRef_(node->getR(), lfr, vr, dr);
    d = dl + dr;
    if (!vl && !lfl) {
      v = vr;
      lf = lfr;
    } else if (!vr && !lfr) {
      v = vl;
      lf = lfl;
    } else {
      lf = (LinearFunctionPtr) new LinearFunction();
      if (lfl) {
        lf->add(lfl);
      } else if (vl) {
        lf->incTerm(vl, 1.0);
      }
      if (lfr) {
        lf->add(lfr);
      } else if (vr) {
        lf->incTerm(vr, 1.0);
      }
      v = 0;
    }
    break;
  case (OpPow):
    assert(!"not implemented!");
    break;
  case (OpPowK):
    recursRef_(node->getL(), lfl, vl, dl);
    powKRef_(lfl, vl, dl, node->getR()->getVal(), lf, v, d);
    break;
  case (OpRound):
  case (OpSin):
  case (OpSinh):
  case (OpSqr):
    recursRef_(node->getL(), lfl, vl, dl);
    uniVarRef_(node, lfl, vl, dl, lf, v, d);
    break;
  case (OpSqrt):
    recursRef_(node->getL(), lfl, vl, dl);
    uniVarRef_(node, lfl, vl, dl, lf, v, d);
    break;
  case (OpSumList):
    d = 0;
    lf = (LinearFunctionPtr) new LinearFunction();
    for (CNode **it=node->getListL(); it!=node->getListR(); ++it) {
      n1 = *it;
      //lfl.reset(); 
      lfl = 0;
      vl = 0; dl = 0;
      recursRef_(n1, lfl, vl, dl);
      d += dl;
      if (lfl) {
        lf->add(lfl);
      } else if (vl) {
        lf->incTerm(vl, 1.0);
      }
    }
    break;
  case (OpTan):
  case (OpTanh):
    recursRef_(node->getL(), lfl, vl, dl);
    uniVarRef_(node, lfl, vl, dl, lf, v, d);
    break;
  case (OpUMinus):
    recursRef_(node->getL(), lfl, vl, dl);
    d = -1.0*dl;
    if (lfl) {
      lf = lfl;
      lf->multiply(-1.0);
    } else if (vl) {
      lf = (LinearFunctionPtr) new LinearFunction();
      lf->addTerm(vl, -1.0);
    }
    break;
  case (OpVar):
    v = newp_->getVariable(node->getV()->getId());
    break;
  default:
    assert(!"cannot evaluate!");
  }

  assert(!lf || !v);
  if (lf && lf->getNumTerms()==1 &&
      fabs(lf->termsBegin()->second-1.0)<zTol_) { // return v, not lf
    v = lf->termsBegin()->first;
    //lf.reset();
    lf = 0;
  }
}


void SimpleTransformer::refNonlinCons_(ConstProblemPtr oldp)
{
  ConstraintPtr c, cnew;
  FunctionPtr f, f2;
  CGraphPtr cg;
  LinearFunctionPtr lf = 0, lf2;
  QuadraticFunctionPtr qf, qf2;
  double d, lb, ub;
  VariablePtr v = VariablePtr();

  assert (oldp && newp_);

  for (ConstraintConstIterator it=oldp->consBegin(); it!=oldp->consEnd();
       ++it) {
    c = *it;
    f = c->getFunction();
    if (f && f->getType()!=Constant && f->getType()!=Linear) {
      lf = f->getLinearFunction();
      if (lf) {
        lf2 = lf->cloneWithVars(newp_->varsBegin());
      } else {
        lf2 = (LinearFunctionPtr) new LinearFunction();
      }
      //lf.reset();
      v = 0; d = 0.0;
      lf = 0;
#if SPEW
      logger_->msgStream(LogDebug) << me_ << "reformulating the constraint"
                                   << std::endl;
      c->write(logger_->msgStream(LogDebug));
#endif
      qf = f->getQuadraticFunction();
      if (qf) {
        qf2 = qf->cloneWithVars(newp_->varsBegin());
        if ((c->getConvexity() == Convex
            && !(f->getNonlinearFunction()))) {
          f2 = (FunctionPtr) new Function(lf2, qf2);
          cnew = newp_->newConstraint(f2, c->getLb(), c->getUb());
          cnew->setConvexity(c->getConvexity());
          cxqHandler_->addConstraint(cnew);
        } else {
          refQuadCons_(qf2, c->getLb(), c->getUb(), lf);
          delete qf2;
        }
      }

      if (f->getNonlinearFunction()) {
        cg = dynamic_cast <CGraph*> (f->getNonlinearFunction());
        assert(cg || qf);
        recursRef_(cg->getOut(), lf, v, d);
      }
      if (lf) {
        lf2->add(lf);
        if (lf2->getNumTerms()>1) {
          f2 = (FunctionPtr) new Function(lf2);
          cnew = newp_->newConstraint(f2, c->getLb()-d, c->getUb()-d);
          lHandler_->addConstraint(cnew);
        } else if (lf2->getNumTerms()==1) {
          v = lf->termsBegin()->first;
          d = lf->termsBegin()->second;
          if (d>0) {
            lb = c->getLb()/d;
            ub = c->getUb()/d;
          } else {
            lb = c->getUb()/d;
            ub = c->getLb()/d;
          }
          if (lb>v->getLb()) {
            newp_->changeBound(v, Lower, lb);
          }
          if (ub<v->getUb()) {
            newp_->changeBound(v, Upper, ub);
          }
#if SPEW
          logger_->msgStream(LogDebug) << me_ << "new bounds on variable "
                                       << std::endl;
          v->write(logger_->msgStream(LogDebug));
#endif 
        } else if ((lf2->getNumTerms()==0) &&
                   (d > c->getUb()+zTol_ ||
                    d < c->getLb()-zTol_)) {
            logger_->msgStream(LogInfo) << me_ << "problem infeasible." << std::endl;
          }
      } else if (v) {
          lf2->incTerm(v, 1.0);
          f2 = (FunctionPtr) new Function(lf2);
          cnew = newp_->newConstraint(f2, c->getLb()-d, c->getUb()-d);
          lHandler_->addConstraint(cnew);
      } 
    } // other case already dealt with in copyLinear_() 
  }
  if (lf) {
    delete lf;
  }
}


void SimpleTransformer::refNonlinObj_(ConstProblemPtr oldp) 
{
  ObjectivePtr obj;
  FunctionPtr f, f2;
  double d = 0;
  VariablePtr v = VariablePtr();
  LinearFunctionPtr lf, lf2;
  QuadraticFunctionPtr qf, qf2;
  CGraphPtr cg;

  assert(newp_);
  assert(oldp);

  obj = oldp->getObjective();
  if (!obj) {
    // already dealt with this case in linearCopy_()
    return;
  }

  f = obj->getFunction();
  if (!f) {
    // already dealt with this case in linearCopy_()
    return;
  }

  if (f->getType()!=Linear && f->getType()!=Constant) {
    lf = f->getLinearFunction();
    if (lf) {
      lf2 = lf->cloneWithVars(newp_->varsBegin());
    } else {
      lf2 = (LinearFunctionPtr) new LinearFunction();
    }
#if SPEW
    logger_->msgStream(LogDebug) << me_ << "reformulating the objective"
      << std::endl;
    obj->write(logger_->msgStream(LogDebug));
#endif
    qf = f->getQuadraticFunction();
    if (qf) {
      qf2 = qf->cloneWithVars(newp_->varsBegin());
      refQuadCons_(qf2, -INFINITY, 0.0, lf);
      d = obj->getConstant();
    }
    
    cg = dynamic_cast <CGraph*> (f->getNonlinearFunction());
    if (cg) {
      recursRef_(cg->getOut(), lf, v, d);
    }
    if (lf) {
      lf2->add(lf);
      if (lf2->getNumTerms()>0) {
        f2 = (FunctionPtr) new Function(lf2);
      } else {
        f2 = FunctionPtr(); // NULL
      }
      obj = newp_->newObjective(f2, d, Minimize);
    } else if (v) {
      lf2->incTerm(v, 1.0);
      if (lf2->getNumTerms()>0) {
        f2 = (FunctionPtr) new Function(lf2);
      } else {
        f2 = FunctionPtr(); // NULL
      }
      obj = newp_->newObjective(f2, d, Minimize);
    } else {
      //f2.reset();
      f2 = 0;
      obj = newp_->newObjective(f2, d, Minimize);
      logger_->msgStream(LogDebug)
        << "Problem objective reduced to a constant" << std::endl;
    }
  } // else the other case is already handled in copyLinear_()
}

bool SimpleTransformer::checkQuadConvexity_(bool& addCxqHandler) {
  QfVector qf_vector;
  QfVector::iterator it;
  bool convex_cons;
  bool all_convex = true;
  ConstraintPtr c;
  QuadraticFunctionPtr qf;
  Convexity sg, sg_old;

  for (ConstraintConstIterator cit=p_->consBegin(); cit!=p_->consEnd(); ++cit) {
    c = *cit;
    qf = c->getFunction()->getQuadraticFunction();
    if (qf) {
      convex_cons = true;
      sg_old = Unknown;
      qf_vector = qf->findSubgraphs();
      for (it = qf_vector.begin(); it != qf_vector.end(); ++it) {
        sg = (*it)->getConvexity();
        if (sg == Nonconvex) {
          convex_cons = false;
          (*it)->setConvexity(Nonconvex);
          c->setConvexity(Nonconvex);
          all_convex = false;
        } else {
          addCxqHandler = true;
          if (sg_old != Unknown && sg != sg_old) {
            convex_cons = false;
            (*it)->setConvexity(Nonconvex);
            c->setConvexity(Nonconvex);
            all_convex = false;
          }
          sg_old = sg;
        }
      }
      if (convex_cons) {
        if (sg == Convex) {
          qf->setConvexity(Convex);
          if (c->getLb() > -INFINITY) {
            if (c->getUb() < INFINITY) {
              c->setConvexity(Nonconvex);
            } else {
              c->setConvexity(Concave);
              all_convex = false;
            }
          } else {
            c->setConvexity(Convex);
          }
        } else {
          qf->setConvexity(Concave);
          if (c->getUb() < INFINITY) {
            if (c->getLb() > -INFINITY) {
              c->setConvexity(Nonconvex);
            } else {
              c->setConvexity(Concave);
              all_convex = false;
            }
          } else {
            c->setConvexity(Convex);
          }
        }
      }
    }
  }
  qf = p_->getObjective()->getFunction()->getQuadraticFunction();
  if (qf) {
    convex_cons = true;
    qf_vector = qf->findSubgraphs();
    for (it = qf_vector.begin(); it != qf_vector.end(); ++it) {
      sg = (*it)->getConvexity();
      if (sg == Nonconvex || sg == Concave) {
        convex_cons = false;
        all_convex = false;
      } 
    }
    if (convex_cons) {
      addCxqHandler = true;
    }
  }
  return all_convex;
}

void SimpleTransformer::refQuadCons_(QuadraticFunctionPtr qf, double lb,
                                     double ub, LinearFunctionPtr &lf)
{
  VariablePtr v = VariablePtr();
  QfVector qf_vector = qf->findSubgraphs();
  LinearFunctionPtr lfnew;
  QuadraticFunctionPtr qfnew;
  FunctionPtr fnew;
  ConstraintPtr cnew;
  double lbnew = 0.0;
  double ubnew = 0.0;
  Convexity cons_cvx = Nonconvex;
  bool flag = false;
  lf = (LinearFunctionPtr) new LinearFunction();

  for (QfVector::iterator qit = qf_vector.begin(); qit != qf_vector.end();
      ++qit) {
    if ((*qit)->getNumTerms() > 1) {
      if ((*qit)->getConvexity() == Convex) {
        if (lb <= -INFINITY) {
          lbnew = lb;
          cons_cvx = Convex;
          flag = true;
        } else if (ub >= INFINITY) {
          ubnew = ub;
          cons_cvx = Concave;
          if ((*qit)->getNumVars() <= 5) {
            flag = true;
          }
        }
      } else if ((*qit)->getConvexity() == Concave) {
        if (ub >= INFINITY) {
          ubnew = ub;
          cons_cvx = Convex;
          flag = true;
        } else if (lb <= -INFINITY) {
          lbnew = lb;
          cons_cvx = Concave;
          if ((*qit)->getNumVars() <= 5) {
            flag = true;
          }
        }
      }
      if (flag) {
        v = yQf_->findY((*qit));
        if (!v) {
          v = newp_->newVariable();
          lfnew = (LinearFunctionPtr) new LinearFunction();
          lfnew->addTerm(v, -1.0);
          fnew = (FunctionPtr) new Function(lfnew, (*qit));
          cnew = newp_->newConstraint(fnew, lbnew, ubnew);
#if SPEW
          logger_->msgStream(LogDebug) << me_ << "added new constraint"
                                              << std::endl;
          cnew->write(logger_->msgStream(LogDebug));
#endif
          cnew->setConvexity(cons_cvx);
          cxqHandler_->addConstraint(cnew);
          yQf_->insert(v, (*qit));
        }
        lf->addTerm(v, 1.0);
        continue;
      } 
    }
    for (VariablePairGroupConstIterator it = (*qit)->begin();
                                        it != (*qit)->end(); ++it) {
      qfnew = (QuadraticFunctionPtr) new QuadraticFunction();
      qfnew->addTerm(it->first.first, it->first.second, 1.0);
      v = yQf_->findY(qfnew);
      if (!v) {
        v = newp_->newVariable();
        lfnew = (LinearFunctionPtr) new LinearFunction();
        lfnew->addTerm(v, -1.0);
        fnew = (FunctionPtr) new Function(lfnew, qfnew);
        cnew = newp_->newConstraint(fnew, lbnew, ubnew);
#if SPEW
        logger_->msgStream(LogDebug) << me_ << "added new constraint"
                                            << std::endl;
        cnew->write(logger_->msgStream(LogDebug));
#endif
        cnew->setConvexity(cons_cvx);
        qHandler_->addConstraint(cnew);
        yQf_->insert(v, qfnew);
      }
      lf->addTerm(v, it->second);
    }
  }
}

void SimpleTransformer::reformulate(ProblemPtr &newp, HandlerVector &handlers,
                                    int &status)
{
  bool addCxqHandler = false;
  assert(p_);

  newp_ = (ProblemPtr) new Problem(env_);
  yLfs_ = new YEqLFs(2*p_->getNumVars());
  yUniExprs_ = new YEqUCGs();
  yBiVars_ = new YEqCGs();
  yVars_ = new YEqVars(p_->getNumVars()+40);
  yQf_ = new YEqQf();
  copyVars_(p_, newp_);

  // create handlers.
  if (p_->getSize()->bins > 0 || p_->getSize()->ints > 0) {
    IntVarHandlerPtr ihandler = (IntVarHandlerPtr)
                                new IntVarHandler(env_, newp_);
    handlers.push_back(ihandler);
  }
  lHandler_ = (LinearHandlerPtr) new LinearHandler(env_, newp_);
  lHandler_->setPreOptPurgeVars(false);
  lHandler_->setPreOptPurgeCons(false);
  lHandler_->setPreOptDualFix(false);
  lHandler_->setModFlags(true, true);
  handlers.push_back(lHandler_);
  qHandler_ = (QuadHandlerPtr) new QuadHandler(env_, newp_);
  qHandler_->setModFlags(true, true);
  handlers.push_back(qHandler_);
  cxqHandler_ = (CxQuadHandlerPtr) new CxQuadHandler(env_, newp_, nlpe_);
  if (checkQuadConvexity_(addCxqHandler)) {
    // TODO : For a convex problem call any convex algorithm.
    // Do not add the handler as well in that case.
    handlers.push_back(cxqHandler_);
  } else {
    if (addCxqHandler) {
      handlers.push_back(cxqHandler_);
    }
  }
  uHandler_ = (CxUnivarHandlerPtr) new CxUnivarHandler(env_, newp_);
  handlers.push_back(uHandler_);

  copyLinear_(p_, newp_);
  refNonlinCons_(p_);
  refNonlinObj_(p_);
  newp_->calculateSize();

#if DEBUG
  assert(0==newp_->checkConVars());
#endif 

  if (!(allConsAssigned_(newp_, handlers))) {
    status = 1;
    return;
  }
  clearUnusedHandlers_(handlers);
  status = 0;
  newp = newp_;
  // newp->write(std::cout);
}


void SimpleTransformer::trigRef_(OpCode op, LinearFunctionPtr lfl,
                                 VariablePtr vl, double dl, VariablePtr &v,
                                 double &d)
{
  if (lfl) {
    vl = newVar_(lfl, dl, newp_);
  } else if (vl && fabs(dl)>zTol_) {
    vl = newVar_(vl, dl, newp_);
  }
  if (vl) {
    CGraphPtr cg = (CGraphPtr) new CGraph();
    CNode *n1 = cg->newNode(vl);
    CNode *n2 = 0;

    n1 = cg->newNode(op, n1, n2);
    cg->setOut(n1);
    cg->finalize();
    v = newVar_(cg, newp_);
  } else {
    d = fabs(dl);
  }
}


void SimpleTransformer::uniVarRef_(const CNode *n0, LinearFunctionPtr lfl,
                                   VariablePtr vl, double dl, 
                                   LinearFunctionPtr &lf, VariablePtr &v,
                                   double &d)
{
  CNode *n1, *n2;
  int err = 0;
  if (lfl) {
    vl = newVar_(lfl, dl, newp_);
    dl = 0.0;
  }
  if (vl) {
    if (fabs(dl)>zTol_) {
      vl = newVar_(vl, dl, newp_);
    }
    CGraphPtr cg = (CGraphPtr) new CGraph();
    n1 = cg->newNode(vl);
    n2 = 0;
    n2 = cg->newNode(n0->getOp(), n1, n2);
    cg->setOut(n2);
    cg->finalize();
    v = 0;
    //lf.reset();
    lf = 0;
    v = newVar_(cg, newp_);
    d = 0;
  } else {
    d = n0->evalSingle(dl, &err);
    assert(0==err);
  }
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
