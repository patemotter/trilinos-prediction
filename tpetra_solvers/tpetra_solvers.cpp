#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
	bool success = false;
	json j;
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
	belosSolvers = belos_all;

    for (int i = 2; i < argc; i++) {
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

	if (myRank == 0) std::cout << "After setup\n";

	const RCP<const MAT> A = Reader::readSparseFile(inputFile, comm, node, true);
	Teuchos::oblackholestream blackhole;
	std::ofstream outputLocJSON, outputLocCSV;

	if (outputDir.size() && outputFile.empty()) {
		if (myRank == 0) {
			unsigned long found = inputFile.find_last_of("/\\");
			std::string outputFilenameJSON = outputDir + "/" + inputFile.substr(found + 1) + ".json";
			std::string outputFilenameCSV = outputDir + "/" + inputFile.substr(found + 1) + ".csv";
			inputFile = inputFile.substr(found + 1);
			outputLocJSON.open(outputFilenameJSON.c_str());
			outputLocCSV.open(outputFilenameCSV.c_str());
		}
		//fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(outputLoc));
	}  
    Teuchos::Time timer("timer", false);
    unsigned long found = inputFile.find_last_of("/\\");
    std::string matrixName = inputFile.substr(found + 1);

	for (std::string solverIter : belos_all) {
    	for (std::string precIter : ifpack2Precs) {
                timer.start(true);
				try {
					if (myRank == 0) {
                        std::cout << "In Belos Solve\n";
					    j[solverIter][precIter]["started"] = true;
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
					prec = ifpack2Factory.create(precIter, A);
					prec->initialize();
					prec->compute();

					//  Create the linear problem w/ prec
					RCP<LP> problem = rcp(new LP(A, x, b));
                    problem->setLeftPrec(prec);
					bool set = problem->setProblem();       // done adding to the linear problem

                    //  Setup belos solver
					RCP<ParameterList> solverParams = parameterList();
                    //solverParams->set("Maximum Iterations", 10000);
                    //solverParams->set("Convergence Tolerance", 1.0e-6);
					Belos::SolverFactory<ST, MV, OP> belosFactory;
					RCP<BSM> solver = belosFactory.create(solverIter, solverParams);
					solver->setProblem(problem); // add the linear problem to the solver

					// Solve
					Belos::ReturnType result = solver->solve();
                    timer.stop();
					if (myRank == 0) {
                        outputLocCSV << matrixName << ", "
                                  << solverIter << ", "
                                  << precIter << ", ";
                        if (result == Belos::Converged)	{
                            j[solverIter][precIter]["solved"] = "converged";
                            outputLocCSV << "converged, "
                                         << timer.totalElapsedTime() << ", "
                                         << solver->getNumIters() << std::endl;
                        } else {
                            j[solverIter][precIter]["solved"] = "unconverged";
                            outputLocCSV << "unconverged, "
                                         << timer.totalElapsedTime() << ", "
                                         << solver->getNumIters() << std::endl;
                        }
						j[solverIter][precIter]["time"] = timer.totalElapsedTime();
						j[solverIter][precIter]["iterations"] = solver->getNumIters();

					}
				} catch (...) {
                    timer.stop();
					if (myRank == 0) {
                        outputLocCSV << matrixName << ", "
                                  << solverIter << ", "
                                  << precIter << ", error\n";
						j[solverIter][precIter]["solved"] = "error";
					}
				}
		}
	}
	if (myRank == 0) {
        outputLocJSON << std::setw(4) << j << "," << std::endl;
        outputLocJSON.close();
        outputLocCSV.close();
    }
}
