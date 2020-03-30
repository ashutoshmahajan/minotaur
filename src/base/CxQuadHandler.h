//
//     MINOTAUR -- It's only 1/2 bull
//
//     (C)opyright 2008 - 2017 The MINOTAUR Team.
//

/**
 * \file CxQuadHandler.h
 * \brief Define the CxQuadHandler class for handling convex quadratic
 * objective functions and constraints.
 * \author Ashutosh Mahajan, Argonne National Laboratory
 */

#ifndef MINOTAURCXQUADRATICHANDLER_H
#define MINOTAURCXQUADRATICHANDLER_H

#include "Handler.h"
#include "Types.h"
#include "Engine.h"

namespace Minotaur {

class Engine;
class Function;
class LinearFunction;
class Problem;
class Timer;
class QuadraticFunction;
typedef LinearFunction* LinearFunctionPtr;
typedef QuadraticFunction* QuadraticFunctionPtr;

struct CxQStats {
  size_t nlpS;      /// Number of nlps solved.
  size_t nlpF;      /// Number of nlps feasible.
  size_t nlpI;      /// Number of nlps infeasible.
  size_t nlpIL;     /// Number of nlps hits engine iterations limit.
  size_t cuts;      /// Number of cuts added to the LP.
};

/**
 * An CxQuadHandler handles the convex parts of quadratic functions of a
 * problem. For now, we will just handle squares of singleton variables e.g.
 * \f$\sum_ix_i^2 \leq u_0\f$. \f$u_0\f$ could be an auxiliary varible or a
 * constant. Later we will introduce sums of squares of
 * general linear functions as well.
 */
class CxQuadHandler : public Handler {
public:
  /// Default constructor.
  CxQuadHandler(EnvPtr env, ProblemPtr problem, EnginePtr nlp_e);

  /// Destroy
  ~CxQuadHandler();

  /// Add a new convex or concave constraint.
  void addConstraint(ConstraintPtr newcon);

  // Does nothing.
  void relaxInitFull(RelaxationPtr rel, bool *is_inf);

  // Does nothing.
  void relaxInitInc(RelaxationPtr rel, bool *is_inf);

  // Does nothing.
  void relaxNodeFull(NodePtr node, RelaxationPtr rel, bool *is_inf);

  // Does nothing.
  void relaxNodeInc(NodePtr node, RelaxationPtr rel, bool *is_inf);


  /**
   * Checking the activity of each constraint in the handler
   * and return false when it violates the first constraint
   * return true if no constraint is violated
   **/
  bool isFeasible(ConstSolutionPtr sol, RelaxationPtr relaxation, 
                  bool &is_inf, double &inf_meas);

  /**
   * Not implemented yet.
   */
  void separate(ConstSolutionPtr sol, NodePtr node, RelaxationPtr rel, 
                CutManager *cutman, SolutionPoolPtr s_pool, ModVector &p_mods,
                ModVector &r_mods, bool *sol_found,
                SeparationStatus *status);


  /// Return \f$u_0\f$ it is constrained to be an integer.
  void getBranchingCandidates(RelaxationPtr rel, 
                              const DoubleVector &x, ModVector &mods,
                              BrVarCandSet &cands, BrCandVector &,
                              bool & is_inf) {};

  // Implement Handler::getBrMod().
  ModificationPtr getBrMod(BrCandPtr cand, DoubleVector &x, 
                           RelaxationPtr rel, BranchDirection dir) {};

  // Implement Handler::getBranches().
  Branches getBranches(BrCandPtr cand, DoubleVector & x,
                       RelaxationPtr rel, SolutionPoolPtr s_pool) {};

  // presolve.
  SolveStatus presolve(PreModQ *pre_mods, bool *changed);

  // Implement Handler::presolveNode().
  bool presolveNode(RelaxationPtr rel, NodePtr node,
                    SolutionPoolPtr s_pool, ModVector &p_mods,
                    ModVector &r_mods);

  void writeStats(std::ostream &out) const;

  // Write name
  std::string getName() const;

private:
  /// Store statistics of separation.
  struct SepaStats 
  {
    int iters;   ///> Number of times separation routine called. 
    int cuts;    ///> Number of cuts added.
    double time; ///> Total time used in separation
  };

  /// Store statistics of presolving.
  struct PresolveStats 
  {
    int iters;   ///> Number of iterations (main cycle).
    double time; ///> Total time used in initial presolve.
    double timeN;///> Total time used in presolveNode.
    int vBnd;    ///> Number of times variable-bounds were tightened.
    int nMods;   ///> Number of changes made in all nodes.
  };

  /// Absolute feasibility tolerance
  double aTol_;

  /// Update bounds if they can be increased by atleastthis quatity
  double bTol_;

  /// Tolerance for constraint violation
  double eTol_;

  /// Relative feasibility tolerance
  double rTol_;

  /// Tolerance when upper and lower are considered equal
  double vTol_;
  
  /// Pointer to the environment
  EnvPtr env_;

  /// Log
  LoggerPtr logger_;

  /// To keep time
  const Timer* timer_;

  /// For log
  static const std::string me_;

  /// Pointer to the original problem
  ProblemPtr problem_;

  /** One sided convex constraints
   * Equality constraint with convex or concave functions
   * Range constraints with convex or concave functions
   */
  ConstraintVector cxCons_;

  /** One sided concave constraints
   * Equality constraint with convex or concave functions
   * Range constraints with convex or concave functions
   */
  ConstraintVector cvCons_;
  
  /// NLP/QP Engine used to solve convex QP relaxation of the problem
  EnginePtr nlpe_;

  /// Status of the NLP/QP engine.
  EngineStatus nlpStatus_;

  /// Statistics about presolve
  PresolveStats pStats_;

  /// Statistics about separation
  SepaStats sStats_;
  
  /// Pointer to relaxation of the problem
  RelaxationPtr rel_;
  
  /// Statistics
  CxQStats *stats_;

  /**
   * Add linearization of nonlinear constraints and objective at point x* 
   * to the relaxation only (not to the lp engine)
   */
  void addInitLinearX_(const double *x);
  
  /// Adds secants to concave constraints
  void addSecant_(QuadraticFunctionPtr qf, Convexity cvx, double rhs,
                  std::vector<BoundType> b, VariablePtr y, DoubleVector evals);

  /// creates a convex NLP relaxation of the problem
  void createConvexRelaxation_();

  /// Delete Quad constraints from the relaxation after solving convex QCQP.
  void deleteQuadConsfromRel_();

  /** For a convex constraint, if the LP solution does not satisfy it
   * then that point will be separated.
   * This function finds a point which is feasible to the constraint and
   * cuts off the LP point
   * Not implemented yet. Might need it in future.
   */
  void findLinPt_(FunctionPtr f, double *x, double ub, double *grad,
                  double *y, Convexity cvx) {};
  /** Solve the convex NLP relaxation of nonconvex MINLP and add linearization
   * about the optimal point. isInf is set to true if the relaxation is found
   * infeasible. Throw an assert if the relaxation is unbounded.
   */
  void initLinear_(bool *isInf); 

  /**
   * Obtain the linear function (lf) and constant (c) from the
   * linearization of function f at point x.
   */
  void linearAt_(FunctionPtr f, double fval, const double *x, 
                 double *c, LinearFunctionPtr *lf, int *error);

  void propBounds_(QuadraticFunctionPtr qf, double &lb, double &ub);

  void propBounds_(FunctionPtr f, double &lb, double &ub);

  void propBounds_(LinearFunctionPtr lf, double &lb, double &ub);

  /// Create Initial Relaxation
  void relax_(RelaxationPtr rel, bool *isInf);

  /// Relaxing concave constraints by adding overestmators.
  void relaxConcaveCons_();

  /// Solve the QCQP.
  void solveQCQP_();

  void resetStats_();

  /**
   * \brief Modify bounds of a variable in the problem to the new bounds lb
   * and ub if the new bounds are tighter.
   * \param[in] v The variable
   * \param[in] lb The new lower bound
   * \param[in] ub The new upper bound
   * \param[out] changed True if the new bounds are tighter than the existing
   * bounds. False otherwise.
   * \return the number of bound changes (1 or 2) if any bounds are changed.
   * Returns -1 if the new bounds make the problem infeasible. Returns 0
   * otherwise.
   */
  int updatePBounds_(VariablePtr v, double lb, double ub, bool *changed);
  
  int updatePBounds_(VariablePtr v, double lb, double ub, RelaxationPtr rel,
                     bool *changed, bool mod_rel, ModVector &p_mods,
                     ModVector &r_mods);
};

/// Shared pointer to CxQuadHandler.
typedef CxQuadHandler* CxQuadHandlerPtr;

/// Shared pointer to const CxQuadHandler.
typedef const CxQuadHandler* CxQuadConstHandlerPtr;
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
