#include "tpetra_properties_crsmatrix.h"

/* 	outputs in order (33): matrix, dimension, frobNorm, symmFrobNorm,
 *	antisymmFrobNorm, nnz, maxNonzerosPerRow, diagonalNNZ, lowerBW, upperBW,
 *	colVariance, rowDiagonalDominance, colDiagonalDominance, diagonalMean,
 *	diagonalSign, diagVariance, numDummyRows, infNorm, symmInfNorm,
 *  antisymmInfNorm, minNNZPerRow, AvgNNZPerRow, oneNorm, rowVariance,
 *	absNonzeroSum, nonzeroSum, matchPercentage, noMatchPercentage,
 *  DNEPercentage, matchBinary, noMatchBinary, DNEBinary
*/

RCP<const Teuchos::Comm<int> > comm;
RCP<Teuchos::FancyOStream> fos;
int myRank, numNodes;


int main(int argc, char *argv[]) {
	std::string outputDir;
	std::vector<std::string> args;
	std::ofstream outputFile;
	json j;
	bool csv = 0, json = 0;
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}

	if (argc < 2) {
		std::cout << "No input file was specified" << std::endl;
		return -1;
	} else 	if (args[2] == "-csv") {
		csv = true;
	} else	if (args[2] == "-json") {
		json = true;
	} else {
		std::cout << "Specify -csv or -json\n";
		exit(-1);
	}
	std::string filename = args[1];

	//  Check if matrix is complex
	std::ifstream infile;
	infile.open(args[1]);
	if (infile.is_open()) {
		std::string firstLine;
		getline(infile, firstLine);
		if (firstLine.find("complex") != std::string::npos) {
			std::cout << "Complex matrices are not currently supported: exiting\n";
			infile.close();
			exit(-1);
		}
	}

	//  General setup for Teuchos/communication
	Teuchos::GlobalMPISession mpiSession(&argc, &argv);
	Platform& platform = Tpetra::DefaultPlatform::getDefaultPlatform();
	comm = platform.getComm();
	RCP<NT> node = platform.getNode();
	myRank = comm->getRank();
	Teuchos::oblackholestream blackhole;
	std::ostream& out = (myRank == 0) ? std::cout : blackhole;
	RCP<MAT> A = Reader::readSparseFile(args[1], comm, node, true);

	//  Decide to print to screen or file
	if (json == 0 && csv == 0) {
		std::cout << "No output directory was specified. Printing to screen\n";
		fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
		unsigned found = filename.find_last_of("/\\");
		*fos << filename.substr(found+1) << CSV;
		runGauntlet(A);
	} else if (json == 0 && csv == 1){ //  print to file
		unsigned found = args[1].find_last_of("/\\");
		outputFile.open(args[3], std::ofstream::app);
		fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(outputFile));
		*fos << filename.substr(found+1) << CSV;
		runGauntlet(A);
	} else if (json == 1 && csv == 0) {
		unsigned found = args[1].find_last_of("/\\");
		outputFile.open(args[3], std::ofstream::app);
		j["matrix_name"] = filename.substr(found+1);
		runGauntlet(A, j, outputFile);
	} else {
		std::cout << "Incorrect flags\n";
		exit(-1);
	}
}


void runGauntlet(const RCP<MAT> &A) {
	// Test squareness
	if (A->getGlobalNumRows() != A->getGlobalNumCols() ) {
		*fos << "Not a square matrix, exiting." << std::endl;
		exit(-1);
	}
	funcsBuiltin(A);
	funcsBandwidth(A);
	calcColVariance(A);
	funcsDiagonalDominance(A);
	calcDiagonalMean(A);
	calcDiagonalSign(A);
	calcDiagVariance(A);
	calcDummyRows(A);
	funcsInfNorm(A);
	funcsNonzeros(A);
	calcOneNorm(A);
	calcRowVariance(A);
	funcsSum(A);
	calcSymmetry(A);
	*fos << std::endl;
}

void runGauntlet(const RCP<MAT> &A, json  &j, std::ofstream &outputFile) {
	// Test squareness
	if (A->getGlobalNumRows() != A->getGlobalNumCols() ) {
		*fos << "Not a square matrix, exiting." << std::endl;
		exit(-1);
	}
	funcsBuiltin(A, j);
	funcsBandwidth(A, j);
	calcColVariance(A, j);
	funcsDiagonalDominance(A, j);
	calcDiagonalMean(A, j);
	calcDiagonalSign(A, j);
	calcDiagVariance(A, j);
	calcDummyRows(A, j);
	funcsInfNorm(A, j);
	funcsNonzeros(A, j);
	calcOneNorm(A, j);
	calcRowVariance(A, j);
	funcsSum(A, j);
	calcSymmetry(A, j);
	outputFile << std::setw(4) << j << std::endl;
}

void funcsBuiltin(const RCP<MAT> &A) {
	calcDim(A);
	calcFrobeniusNorm(A);
	calcSymmetricFrobeniusNorm(A);
	calcAntisymmetricFrobeniusNorm(A);
	calcNonzeros(A);
	calcMaxNonzerosPerRow(A);
	calcDiagonalNonzeros(A);
}
void funcsBuiltin(const RCP<MAT> &A, json &j) {
	calcDim(A, j);
	calcFrobeniusNorm(A, j);
	calcSymmetricFrobeniusNorm(A, j);
	calcAntisymmetricFrobeniusNorm(A, j);
	calcNonzeros(A, j);
	calcMaxNonzerosPerRow(A, j);
	calcDiagonalNonzeros(A, j);
}

void funcsBandwidth(const RCP<MAT> &A) {
	calcLowerBandwidth(A);
	calcUpperBandwidth(A);
}
void funcsBandwidth(const RCP<MAT> &A, json &j) {
	calcLowerBandwidth(A, j);
	calcUpperBandwidth(A, j);
}

void funcsDiagonalDominance(const RCP<MAT> &A) {
	calcRowDiagonalDominance(A);
	calcColDiagonalDominance(A);
}
void funcsDiagonalDominance(const RCP<MAT> &A, json &j) {
	calcRowDiagonalDominance(A, j);
	calcColDiagonalDominance(A, j);
}

void funcsInfNorm(const RCP<MAT> &A) {
	calcInfNorm(A);
	calcSymmetricInfNorm(A);
	calcAntisymmetricInfNorm(A);
}
void funcsInfNorm(const RCP<MAT> &A, json &j) {
	calcInfNorm(A, j, 0);
	calcSymmetricInfNorm(A, j);
	calcAntisymmetricInfNorm(A, j);
}

void funcsNonzeros(const RCP<MAT> &A) {
	calcMinNonzerosPerRow(A);
	calcAvgNonzerosPerRow(A);
}
void funcsNonzeros(const RCP<MAT> &A, json &j) {
	calcMinNonzerosPerRow(A, j);
	calcAvgNonzerosPerRow(A, j);
}

void funcsSum(const RCP<MAT> &A) {
	calcAbsNonzeroSum(A);
	calcNonzeroSum(A);
}
void funcsSum(const RCP<MAT> &A, json &j) {
	calcAbsNonzeroSum(A, j);
	calcNonzeroSum(A, j);
}
