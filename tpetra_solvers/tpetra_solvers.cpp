#include "tpetra_solvers.h"

int symm = 0;

int main(int argc, char *argv[]) {
    bool success = false;
    json j;
    std::string outputDir, outputFile;
    std::string inputFile = argv[1];
    belosSolvers = belos_all;

    Teuchos::GlobalMPISession mpiSession(&argc, &argv);
    Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
    comm = platform.getComm();
    RCP<NT> node = platform.getNode();
    myRank = comm->getRank();

    std::cout << "After setup\n";

    RCP<MAT> A = Reader::readSparseFile(inputFile, comm, node, true);
    Teuchos::oblackholestream blackhole;
    std::ostream &out = (myRank == 0) ? std::cout : blackhole;
    std::ofstream outputLoc;
    //  How to output results
    if (outputDir.empty() && outputFile.empty()) {
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
    } else if (outputDir.empty() && outputFile.size()) {
        if (myRank == 0) {
            std::cout << "Printing to " << outputFile << std::endl;
            outputLoc.open(outputFile.c_str());
        }
    }
    std::cout << "Before barriers\n";
    // Do all the work
    belosSolve(A, inputFile, j);
}

//  https://code.google.com/p/trilinos/wiki/Tpetra_Belos_CreateSolver
void belosSolve(const RCP<const MAT> &A, const std::string &inputFile, json &j) {

    for (auto precIter : ifpack2Precs) {
        for (auto solverIter : belos_all) {
            try {
                std::cout << "In Belos Solve\n";

                // Preconditioner
                Ifpack2::Factory ifpack2Factory;
                RCP<PRE> prec; 
                RCP<LP> problem;
                prec = ifpack2Factory.create(precIter, A);
                prec->initialize();
                prec->compute();

                std::cout << "After prec\n";

                // Solver
                Belos::SolverFactory<ST, MV, OP> belosFactory;
                RCP<ParameterList> solverParams = parameterList();
                RCP<BSM> solver = belosFactory.create(solverIter, solverParams);
                RCP<MV> x = rcp(new MV(A->getDomainMap(), 1));
                RCP<MV> b = rcp(new MV(A->getRangeMap(), 1));
                b->randomize();

                std::cout << "After solver\n";

                //  Create the linear problem
                problem = rcp(new LP(A, x, b));
                problem->setProblem(); //done adding to the linear problem
                solver->setProblem(problem); //add the linear problem to the solver

                std::cout << "After LP\n";

                // Solve
                Belos::ReturnType result = solver->solve();
                std::cout << "After solve\n";
            } catch (...) {
                std::cout << "Catch\n";
            }
        }
    }
}
