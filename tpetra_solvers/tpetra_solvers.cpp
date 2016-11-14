#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
  if (strcmp(argv[1], "-h") == 0) {
    std::cout << "Usage:\n mpirun tpetra_solver <mtx_file> <solver name> "
              << "<preconditioner_name> <number_of_solves> <-d> <output_dir>\n";
    std::cout << "Preconditioner choices:\nRILUK, ILUT, DIAGONAL, CHEBYSHEV, "
              << "BLOCK_RELAXATION, RELAXATION, and SCHWARZ\n";
    std::cout
        << "Solver choices:\nMINRES, PSEUDOBLOCK_CG, PSEUDOBLOCK_STOCHASTIC_CG,"
        << "FIXED_POINT, PSEUDOBLOCK_TFQMR, TFQMR, BICGSTAB, LSQR, "
           "PSEUDOBLOCK_GMRES\n";
  }

  std::string outputDir, outputFile;
  std::string inputFile = argv[1];

  //  Check if matrix is complex or integer
  std::ifstream infile;
  infile.open(inputFile);
  if (infile.is_open()) {
    std::string firstLine;
    getline(infile, firstLine);
    if (firstLine.find("complex") != std::string::npos) {
      std::cout << "Complex matrices are not currently supported: exiting\n";
      infile.close();
      exit(-1);
    }
    if (firstLine.find("integer") != std::string::npos) {
      std::cout << "Integer matrices are not currently supported: exiting\n";
      infile.close();
      exit(-1);
    }
  }

  for (int i = 2; i < argc; i++) {
    std::cout << argv[i] << std::endl;
    if (strcmp(argv[i], "-d") == 0) {
      i++;
      outputDir = argv[i];
    } else {
      std::cout << "not using -d\n";
      exit(-1);
    }
  }

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
  comm = platform.getComm();
  RCP<NT> node = platform.getNode();
  myRank = comm->getRank();

  const RCP<const MAT> A = Reader::readSparseFile(inputFile, comm, node, true);
  Teuchos::oblackholestream blackhole;
  std::ofstream outputLocCSV;

  if (outputDir.size() && outputFile.empty()) {
    if (myRank == 0) {
      unsigned long found = inputFile.find_last_of("/\\");
      std::string outputFilenameCSV = outputDir + "/" + "results.csv";
      inputFile = inputFile.substr(found + 1);
      std::cout << outputFilenameCSV << std::endl;
      outputLocCSV.open(outputFilenameCSV.c_str(),
                        std::ofstream::out | std::ofstream::app);
    }
  }

  for (std::string solverChoice : belos_all) {
    for (std::string precChoice : ifpack2Precs) {
      Teuchos::Time timer("timer", false);
      unsigned long found = inputFile.find_last_of("/\\");
      std::string matrixName = inputFile.substr(found + 1);

      timer.start(true);
      if (myRank == 0) {
        std::cout << "Working on: " << solverChoice << "\t" << precChoice
                  << std::endl;
      }

      // Vectors
      RCP<MV> x = rcp(new MV(A->getDomainMap(), 1));
      RCP<MV> b = rcp(new MV(A->getDomainMap(), 1));
      x->randomize();
      A->apply(*x, *b);
      x->putScalar(0.0);

      // Preconditioner
      Ifpack2::Factory ifpack2Factory;
      RCP<PRE> prec;
      prec = ifpack2Factory.create(precChoice, A);
      prec->initialize();
      prec->compute();

      //  Create the linear problem w/ prec
      RCP<LP> problem = rcp(new LP(A, x, b));
      problem->setLeftPrec(prec);
      problem->setProblem(); // done adding to the linear problem

      //  Setup belos solver
      RCP<ParameterList> solverParams = parameterList();
      solverParams->set("Maximum Iterations", 10000);
      solverParams->set("Convergence Tolerance", 1.0e-6);
      Belos::SolverFactory<ST, MV, OP> belosFactory;
      RCP<BSM> solver = belosFactory.create(solverChoice, solverParams);
      solver->setProblem(problem); // add the linear problem to the solver

      // Solve
      bool error = false;
      Belos::ReturnType result;
      {
        try {
          result = solver->solve();
        } catch (...) {
          timer.stop();
          outputLocCSV << matrixName << ", " << solverChoice << ", "
                       << precChoice << ", error\n";
          error = true;
        }
      }
      timer.stop();
      if (myRank == 0 && error == 0) {
        outputLocCSV << matrixName << ", " << solverChoice << ", " << precChoice
                     << ", ";
        if (result == Belos::Converged) {
          outputLocCSV << "converged, " << timer.totalElapsedTime() << ", "
                       << solver->getNumIters() << std::endl;
        } else {
          outputLocCSV << "unconverged, " << timer.totalElapsedTime() << ", "
                       << solver->getNumIters() << std::endl;
        }
      }
    }
  }
  if (myRank == 0) {
    outputLocCSV.close();
  }
}
