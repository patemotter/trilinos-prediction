//  Tpetra
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_CrsMatrix.hpp>

//  Belos
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

//  Ifpack2
#include <Ifpack2_Factory.hpp>

//  Teuchos
#include <Teuchos_StandardCatchMacros.hpp>

//  c++
#include <exception>
#include <fstream>

//  Typedefs
typedef double ST;
typedef int LO;
typedef int GO;
typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef Tpetra::Map<>::node_type NT;
typedef Tpetra::CrsMatrix<ST, LO, GO, NT> MAT;
typedef Tpetra::MultiVector<ST, LO, GO, NT> MV;
typedef Tpetra::MatrixMarket::Reader<MAT> Reader;
typedef Tpetra::Operator<ST, LO, GO, NT> OP;
typedef Ifpack2::Preconditioner<ST, LO, GO, NT> PRE;
typedef Belos::LinearProblem<ST, MV, OP> LP;
typedef Belos::SolverManager<ST, MV, OP> BSM;
typedef const std::vector<std::string> STRINGS;

//  Namespaces
using Tpetra::global_size_t;
using Tpetra::Map;
using Tpetra::Import;
using Teuchos::RCP;
using Teuchos::rcpFromRef;
using Teuchos::ArrayView;
using Teuchos::Array;
using Teuchos::Time;
using Teuchos::TimeMonitor;
using Teuchos::ParameterList;
using Teuchos::parameterList;

//  Globals
int myRank;
RCP<Teuchos::FancyOStream> fos;
RCP<const Teuchos::Comm<int>> comm;
std::vector<std::string> belosSolvers;

//  6 preconditioners, 7 solvers, 42 combinations
STRINGS ifpack2Precs = {"ILUT",
                        "RILUK", 
                        "RELAXATION", 
                        "CHEBYSHEV",
                        "NONE"}; 

//                      "BLOCK RELAXATION", 
//                      "DIAGONAL"

STRINGS belos_all = {"FIXED POINT", 
                     "BICGSTAB",
                     "MINRES",
                     "PSEUDOBLOCK CG",
                     "PSEUDOBLOCK TFQMR",
                     "LSQR",
                     "PSEUDOBLOCK GMRES"}; 

//                   "TFQMR",
//                   "PSEUDOBLOCK STOCHASTIC CG",
//                   "BICGSTAB",
//                   "BLOCK GMRES",
//                   "FLEXIBLE GMRES",
//                   "RECYCLING CG",
//                   "RECYCLING GMRES",
//                   "SEED CG",
//                   "SEED GMRES",
