//  Tpetra
#include <Tpetra_CrsMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>

//  Belos
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

//  Ifpack2
#include <Ifpack2_Factory.hpp>

//  Teuchos
#include "Teuchos_StandardCatchMacros.hpp"

//  c++
#include <exception>
#include "json.hpp"

//  Typedefs
typedef double ST;
typedef int LO;
typedef int64_t GO;
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
using nlohmann::json;

//  Globals
int myRank;
RCP<Teuchos::FancyOStream> fos;
RCP<const Teuchos::Comm<int> > comm;
std::vector<std::string> belosSolvers;

//  6 precs, 14 solvers, 84 combinations (incl no prec)
STRINGS ifpack2Precs = {"ILUT", "RILUK",/* "DIAGONAL",*/ "RELAXATION", "CHEBYSHEV", "None"};

STRINGS belos_sq = {"PSEUDOBLOCK TFQMR", "TFQMR", "BICGSTAB", "BLOCK GMRES",
                    "PSEUDOBLOCK GMRES", "HYBRID BLOCK GMRES", "GCRODR"}; //, "LSQR"};

STRINGS belos_all = {"PSEUDOBLOCK TFQMR", "BICGSTAB", "BLOCK GMRES",
                     "PSEUDOBLOCK GMRES", "GCRODR", //"LSQR",
                     "BLOCK CG", "PSEUDOBLOCK CG", "PSEUDOBLOCK STOCHASTIC CG",
                     "RCG", "TFQMR", "PCPG", "MINRES", "HYBRID BLOCK GMRES"};

//  Functions
STRINGS determineSolvers(const std::string &filename);

void belosSolve(const RCP<const MAT> &A, const std::string &filename, json &j);

RCP<PRE> getIfpack2Preconditoner(const RCP<const MAT> &A, std::string ifpack2PrecChoice);
