#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
    bool success = false;
    json j;
    std::string outputDir, outputFile;
    if (argv[1] == NULL) {
        std::cout << "No input file was specified" << std::endl;
        std::cout << "Usage: ./tpetra_solvers <.mtx file> ['-d output_dir' | '-f output_file']" << std::endl;
        return -1;
    }
    for (int i = 2; i < argc; i++) {
        //std::cout << "arg[" << i << "]: " << argv[i] << std::endl;
        if (strcmp(argv[i], "-f") == 0) { //output to file
            i++;
            outputFile = argv[i];
        } else if (strcmp(argv[i], "-d") == 0) {
            i++;
            outputDir = argv[i];
        }
    }

    //  Check if matrix is complex or integer
    std::ifstream infile;
    infile.open(argv[1]);
    if (infile.is_open()) {
        std::string firstLine;
        getline(infile, firstLine);
        if (firstLine.find("complex") != std::string::npos || 
                firstLine.find("integer") != std::string::npos) {
            std::cout << "Invalid matrix (complex or integer)\n";
            infile.close();
            exit(-1); 
        }
    }

    //  General setup for Teuchos/communication
    std::string inputFile = argv[1];
    belosSolvers = determineSolvers(inputFile);
    Teuchos::GlobalMPISession mpiSession(&argc, &argv);
    Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
    comm = platform.getComm();
    RCP<NT> node = platform.getNode();
    myRank = comm->getRank();

    RCP<MAT> A = Reader::readSparseFile(inputFile, comm, node, true);
    Teuchos::oblackholestream blackhole;
    std::ostream &out = (myRank == 0) ? std::cout : blackhole;
    std::ofstream outputLoc;
    try {

        //  How to output results
        if (outputDir.empty() && outputFile.empty()) {
            // Print to screen
            //fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
            //*fos << "No output directory or file was specified. Printing to screen" << std::endl;
            unsigned long found = inputFile.find_last_of("/\\");
            inputFile = inputFile.substr(found + 1);
        } else if (outputDir.size() && outputFile.empty()) {
            // Print to directory
            unsigned long found = inputFile.find_last_of("/\\");
            std::string outputFilename = outputDir + "/" + inputFile.substr(found + 1) + ".json";
            if (myRank == 0) {
                std::cout << "Printing to " << outputFilename << std::endl;
                inputFile = inputFile.substr(found + 1);
                outputLoc.open(outputFilename.c_str());
            }
            //fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(outputLoc));
        } else if (outputDir.empty() && outputFile.size()) {
            if (myRank == 0) {
                std::cout << "Printing to " << outputFile << std::endl;
                outputLoc.open(outputFile.c_str());
            }
            //fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(outputLoc));
        }
        // Do all the work
        belosSolve(A, inputFile, j);
        std::string matrixName = inputFile.substr(inputFile.find_last_of("/") + 1);
        if (myRank == 0) {
            j[matrixName]["status"] = "success";
	        outputLoc << std::setw(4) << j << "," << std::endl;
            outputLoc.close();
        }
    } catch (...) {
        if (myRank == 0) {
            std::string matrixName = inputFile.substr(inputFile.find_last_of("/") + 1);
            j[matrixName]["status"] = "failure";
	        outputLoc << std::setw(4) << j << "," << std::endl;
            outputLoc.close();
        }
    }
}

RCP<PRE> getIfpack2Preconditoner(const RCP<const MAT> &A,
        std::string ifpack2PrecChoice) {
    RCP<PRE> prec;
    Ifpack2::Factory ifpack2Factory;
    prec = ifpack2Factory.create(ifpack2PrecChoice, A);
    prec->initialize();
    prec->compute();
    return prec;
}

RCP<BSM> getBelosSolver(const RCP<const MAT> &A,
        std::string belosSolverChoice) {
    Belos::SolverFactory<ST, MV, OP> belosFactory;
    RCP<ParameterList> solverParams = parameterList();
    Teuchos::Array<std::string> s = belosFactory.supportedSolverNames();
    //*fos << s << std::endl;
    solverParams->set("Verbosity", 0);
    //solverParams->set ("Num Blocks", 40);
    solverParams->set("Maximum Iterations", 10000);
    solverParams->set("Convergence Tolerance", 1.0e-5);
    RCP<BSM> solver = belosFactory.create(belosSolverChoice, solverParams);
    return solver;
}

//  https://code.google.com/p/trilinos/wiki/Tpetra_Belos_CreateSolver
void belosSolve(const RCP<const MAT> &A, const std::string &inputFile, json &j) {
    // Create solver and preconditioner 
    Teuchos::Time timer("timer", false);
    Teuchos::Time overall_timer("overall_timer", false);
    RCP<PRE> prec;
    RCP<BSM> solver;
    RCP<MV> origVector = rcp(new MV(A->getRangeMap(), 1));
    origVector->randomize();

    std::string matrixName = inputFile.substr(inputFile.find_last_of("/") + 1);

    overall_timer.start(true);
    //  Solving linear system with all prec/solver pairs
    for (auto precIter : ifpack2Precs) {
        for (auto solverIter : belos_all) {
            timer.start(true);
            solver = Teuchos::null;
            prec = Teuchos::null;
            //*fos << inputFile << ", " << comm->getSize() << ", ";
            //j[inputFile][solverIter][precIter]["matrix_name"] = inputFile;
            if (myRank == 0)
                j[matrixName][solverIter][precIter]["num_procs"] = comm->getSize();
            try {
                solver = getBelosSolver(A, solverIter);
                if (precIter.compare("None"))
                    prec = getIfpack2Preconditoner(A, precIter);
            } catch (const std::exception &exc) {
                //*fos << solverIter << ", " << precIter << ", PREC-SOLVER_ERROR, ";
                //*fos << timer.totalElapsedTime() << std::endl;
                if (myRank == 0) {
                    j[matrixName][solverIter][precIter]["status"] = "prec/solver_error";
                    std::cerr << exc.what() << std::endl;
                }
                break;
            }
            try {
                //  Create the x and randomized b multivectors
                RCP<MV> x = rcp(new MV(A->getDomainMap(), 1));
                origVector->randomize();
                RCP<MV> b = origVector;

                //  Create the linear problem
                RCP<LP> problem = rcp(new LP(A, x, b));
                if (precIter.compare("None"))
                    problem->setRightPrec(prec);
                problem->setProblem(); //done adding to the linear problem
                solver->setProblem(problem); //add the linear problem to the solver
            } catch (const std::exception &exc) {
                //*fos << solverIter << ", " << precIter << ", CREATION_ERROR, ";
                //*fos << timer.totalElapsedTime() << std::endl;
                //j["solver"] = solverIter;
                //j["preconditioner"] = precIter;
                //j["status"] = "creation_error";
                if (myRank == 0) {
                    j[matrixName][solverIter][precIter]["status"] = "creation_error";
                    std::cerr << exc.what() << std::endl;
                }
                break;
            }
            try {
                //  Solve the linear problem 
                Belos::ReturnType result = solver->solve();
                timer.stop();
                //*fos << std::string(solverIter) << ", " << precIter; // output solver/prec pair
                if (result == Belos::Converged) {
                    //*fos << ", converged, ";
                    if (myRank == 0)
                        j[matrixName][solverIter][precIter]["status"] = "converged";
                } else {
                    //*fos << ", unconverged, ";
                    if (myRank == 0)
                        j[matrixName][solverIter][precIter]["status"] = "unconverged";
                }
                //*fos << solver->getNumIters() << ", " << timer.totalElapsedTime() << std::endl;
                //j["solver"] = solverIter;
                //j["preconditioner"] = precIter;
                //j["iterations"] = solver->getNumIters();
                if (myRank == 0) {
                    j[matrixName][solverIter][precIter]["time"] = timer.totalElapsedTime();
                    j[matrixName][solverIter][precIter]["iterations"] = solver->getNumIters();
                }
                //j["time"] = timer.totalElapsedTime();
            } catch (const std::exception &exc) {
                //*fos << solverIter << ", " << precIter << ", SOLVING_ERROR, ";
                //*fos << timer.totalElapsedTime() << std::endl;
                //j["solver"] = solverIter;
                //j["preconditioner"] = precIter;
                //j["status"] = "solving_error";
                if (myRank == 0) {
                    j[matrixName][solverIter][precIter]["status"] = "solving_error";
                    std::cerr << exc.what() << std::endl;
                }
                break;
            }
        }
    }
}

STRINGS determineSolvers(const std::string &inputFile) {
    std::ifstream file(inputFile);
    std::string firstLine, firstNumbers;
    unsigned int rows, cols;
    if (file.good()) {
        std::getline(file, firstLine);
        std::getline(file, firstNumbers);
        while (firstNumbers.find("%") == 0) {
            std::getline(file, firstNumbers);
        }
        std::stringstream ss(firstNumbers);
        ss >> rows >> cols;
    }
    file.close();
    /*
    if (firstLine.find("symmetric") != std::string::npos) { // include all
        symm = 1;
        return belos_all;
    } else if (firstLine.find("general") != std::string::npos) { // only include sq+rec
        symm = 0;
        return belos_sq;
    } else {
        //  Should never be here
        exit(-1);
    }
    */
    return belos_all;
}
