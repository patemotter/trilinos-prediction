#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
	bool success = false;
	json j;
	std::string outputDir, outputFile;
	std::string inputFile = argv[1];
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

	for (std::string precIter : ifpack2Precs) {
		for (std::string solverIter : belos_all) {
			comm->barrier();
                timer.start(true);
				try {
					if (myRank == 0) {
                        std::cout << "In Belos Solve\n";
					    j[solverIter][precIter]["started"] = true;
                    }

					RCP<LP> problem;

					// Preconditioner
					RCP<PRE> prec;
					Ifpack2::Factory ifpack2Factory;
					prec = ifpack2Factory.create(precIter, A);
					prec->initialize();
					prec->compute();

					if (myRank == 0) std::cout << "After prec\n";

					// Solver
					Belos::SolverFactory<ST, MV, OP> belosFactory;
					RCP<ParameterList> solverParams = parameterList();
                    solverParams->set("Maximum Iterations", 10000);
                    solverParams->set("Convergence Tolerance", 1.0e-6);
					RCP<BSM> solver = belosFactory.create(solverIter, solverParams);
					RCP<MV> x = rcp(new MV(A->getDomainMap(), 1));
					RCP<MV> b = rcp(new MV(A->getRangeMap(), 1));
					b->randomize();

					if (myRank == 0) std::cout << "After solver\n";

					//  Create the linear problem
					problem = rcp(new LP(A, x, b));
					problem->setProblem();       // done adding to the linear problem
					solver->setProblem(problem); // add the linear problem to the solver

					if (myRank == 0) std::cout << "After LP\n";

					// Solve
					Belos::ReturnType result = solver->solve();
                    timer.stop();
					if (myRank == 0) {
                        outputLocCSV << matrixName << ", "
                                  << solverIter << ", "
                                  << precIter << ", "
                                  << timer.totalElapsedTime() << ", "
                                  << solver->getNumIters() << std::endl;
                        if (result)	j[solverIter][precIter]["solved"] = "success";
                        else j[solverIter][precIter]["solved"] = "failed";
						j[solverIter][precIter]["time"] = timer.totalElapsedTime();
						j[solverIter][precIter]["iterations"] = solver->getNumIters();
						outputLocJSON << std::setw(4) << j << "," << std::endl;
						//std::cout << "After solve\n";
					}
				} catch (...) {
                    timer.stop();
					if (myRank == 0) {
                        outputLocCSV << matrixName << ", "
                                  << solverIter << ", "
                                  << precIter << ", error\n";
						j[solverIter][precIter]["solved"] = "error";
						j[solverIter][precIter]["time"] = timer.totalElapsedTime();
						outputLocJSON << std::setw(4) << j << "," << std::endl;
						//std::cout << "Catch\n";
					}
				}
		}
	}
    outputLocJSON.close();
    outputLocCSV.close();
}
