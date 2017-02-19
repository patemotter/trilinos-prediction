#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
    if (argc == 1 || strcmp(argv[1], "-h") == 0) {
        std::cout << "Usage:\n mpirun tpetra_solver <mtx_file> <-d> <output_dir>\n";
        std::cout << "Preconditioner choices:\nRILUK, ILUT, CHEBYSHEV, "
                     "BLOCK_RELAXATION, RELAXATION, and SCHWARZ\n";
        std::cout << "Solver choices:\nMINRES, PSEUDOBLOCK_CG, "
                     "PSEUDOBLOCK_STOCHASTIC_CG,"
                     "FIXED_POINT, PSEUDOBLOCK_TFQMR, BICGSTAB, LSQR, "
                     "PSEUDOBLOCK_GMRES\n";
        exit(0);
    }

    std::string outputDir, outputFile;
    std::string inputFile = argv[1];

    //  Check if matrix is complex, integer, or a pattern
    std::ifstream infile;
    infile.open(inputFile);
    if (!infile.good()) {
        std::cout << "Input file (" << inputFile << ") does not exist. Exiting.\n";
        exit(0);
    } else if (infile.is_open()) {
        std::string firstLine;
        getline(infile, firstLine);
        if (firstLine.find("complex") != std::string::npos) {
            std::cout << inputFile
                      << ": Complex matrices are not currently supported. Exiting\n";
            infile.close();
            exit(0);
        }
        if (firstLine.find("integer") != std::string::npos) {
            std::cout << inputFile
                      << ": Integer matrices are not currently supported. Exiting\n";
            infile.close();
            exit(0);
        }
        if (firstLine.find("pattern") != std::string::npos) {
            std::cout << inputFile
                      << ": Pattern matrices are not currently supported: exiting\n";
            infile.close();
            exit(0);
        }
    }

    //  Check for output directory
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            i++;
            outputDir = argv[i];
        } else {
            std::cout << "not using -d for output\n";
            exit(0);
        }
    }

    //  Basic Teuchos MPI setup
    Teuchos::GlobalMPISession mpiSession(&argc, &argv);
    Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
    comm = platform.getComm();
    RCP<NT> node = platform.getNode();
    myRank = comm->getRank();
    int numProcs = comm->getSize();

    //  Read in from .mtx file
    const RCP<const MAT> A = Reader::readSparseFile(inputFile, comm, node, true);
    if (A->getGlobalNumRows() != A->getGlobalNumCols()) {
        if (myRank == 0) {
            std::cout << "Matrix is not square, exiting\n";
        }
        exit(0);
    }

    //  Locations to output to
    Teuchos::oblackholestream blackhole;
    std::ostream &out = (myRank == 0 ? std::cout : blackhole);
    std::ofstream outputLocCSV;

    //  Generate CSV file name based on passed in info
    if (outputDir.size() && outputFile.empty()) {
        if (myRank == 0) {
            unsigned long found = inputFile.find_last_of("/\\");
            inputFile = inputFile.substr(found + 1);
            std::string outputFilenameCSV =
                outputDir + "/results_" + std::to_string(numProcs) + ".csv";
            outputLocCSV.open(outputFilenameCSV.c_str(),
                              std::ofstream::out | std::ofstream::app);
        }
    }

    //  Main loop over each solver and preconditioner
    for (std::string solverChoice : belos_all) {
        for (std::string precChoice : ifpack2Precs) {
            Teuchos::Time timer("timer", false);
            timer.start(true);
            unsigned long found = inputFile.find_last_of("/\\");
            std::string matrixName = inputFile.substr(found + 1);

            if (myRank == 0) {
                std::cout << "\nWorking on: " << matrixName << "\t" << solverChoice
                          << "\t" << precChoice << std::endl;
            }

            // Create and initialize vectors
            RCP<MV> x = rcp(new MV(A->getDomainMap(), 1));
            RCP<MV> b = rcp(new MV(A->getDomainMap(), 1));
            x->putScalar(0.0);
            b->putScalar(1.0);

            //  Setup belos solver
            RCP<ParameterList> solverParams = parameterList();
            solverParams->set("Maximum Iterations", 10000);
            solverParams->set("Convergence Tolerance", 1.0e-6);
            solverParams->set("Verbosity",
                              Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
            //  Fixes issue of GMRES only restarting 20x, stopping at 6300 iters
            if (solverChoice == "PSEUDOBLOCK GMRES") {
                solverParams->set("Maximum Restarts", 100);
            }

            //  Creating the needed solver type w/ changed parameters
            Belos::SolverFactory<ST, MV, OP> belosFactory;
            RCP<BSM> solver = belosFactory.create(solverChoice, solverParams);

            // Creating the needed preconditioner type
            Ifpack2::Factory ifpack2Factory;
            RCP<PRE> prec;
            if (precChoice != "NONE") {
                prec = ifpack2Factory.create(precChoice, A);
                prec->initialize();
                prec->compute();
            } else {
                prec = Teuchos::null;
            }

            //  Create the linear problem
            RCP<LP> problem = rcp(new LP(A, x, b));
            if (precChoice != "NONE") {
                if (solverChoice != "FIXED POINT") { // No right prec exists yet
                    problem->setRightPrec(prec);
                } else {
                    problem->setLeftPrec(prec);
                }
            } else {
                problem->setRightPrec(Teuchos::null);
                problem->setLeftPrec(Teuchos::null);
            }
            problem->setProblem(); // done adding to the linear problem
            solver->setProblem(problem);

            // Solve the problem
            bool error = false;
            Belos::ReturnType result;
            {
                try {
                    result = solver->solve();
                } catch (const std::exception &e) {
                    error = true;
                    std::cerr << e.what();
                }
            }
            timer.stop();
            TimeMonitor::summarize();     // Print timing info to cout
            TimeMonitor::zeroOutTimers(); // Reset timers for next solver-prec

            //  Output results to the csv file
            if (myRank == 0) {
                if (error == false) {
                    outputLocCSV << matrixName << ", " << solverChoice << ", "
                                 << precChoice << ", ";
                    if (result == Belos::Converged) {
                        outputLocCSV << "converged, " << timer.totalElapsedTime() << ", "
                                     << solver->getNumIters() << ", ";
                    } else {
                        outputLocCSV << "unconverged, " << timer.totalElapsedTime()
                                     << ", " << solver->getNumIters() << ", ";
                    }
                    try {
                        outputLocCSV << solver->achievedTol() << std::endl;
                    } catch (...) {
                        outputLocCSV << std::endl;
                    }
                } else {
                    outputLocCSV << matrixName << ", " << solverChoice << ", "
                                 << precChoice << ", error, " << timer.totalElapsedTime()
                                 << std::endl;
                }
            }
        }
    }
    if (myRank == 0) {
        outputLocCSV.close();
    }
}
