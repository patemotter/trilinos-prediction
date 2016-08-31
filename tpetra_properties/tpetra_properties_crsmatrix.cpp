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
	if (argv[1] == NULL) {
		std::cout << "No input file was specified" << std::endl;
		return -1;
	}
	if (argv[2] != NULL) {
		outputDir = argv[2];
	}
	std::string origFilename = argv[1];
	std::string filename = origFilename;


	//  General setup for Teuchos/communication
	Teuchos::GlobalMPISession mpiSession(&argc, &argv);
	Platform& platform = Tpetra::DefaultPlatform::getDefaultPlatform();
	comm = platform.getComm();
	RCP<NT> node = platform.getNode();
	myRank = comm->getRank();
	Teuchos::oblackholestream blackhole;
	std::ostream& out = (myRank == 0) ? std::cout : blackhole;
	std::ofstream outputFile;

	//  Decide to print to screen or file
	if (outputDir.empty()) {
		std::cout << "No output directory was specified. Printing to screen" << std::endl;
		fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
		unsigned found = filename.find_last_of("/\\");
		filename = filename.substr(found+1);
	} else { //  print to file
		unsigned found = filename.find_last_of("/\\");
		std::string outputFilename = outputDir + "/" + filename.substr(found+1)+".out";
		filename = filename.substr(found+1);
		outputFile.open(outputFilename.c_str());
		fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(outputFile));
	}

	//  Check if matrix is complex
	std::ifstream infile;
	infile.open(argv[1]);
	if (infile.is_open()) {
		std::string firstLine;
		getline(infile, firstLine);
		if (firstLine.find("complex") != std::string::npos) {
			*fos << "Complex matrices are not currently supported: exiting\n";
			infile.close();
			exit(-1);
		}
	}
	RCP<MAT> A = Reader::readSparseFile(origFilename, comm, node, true);
	*fos << filename << CSV;
	try {
		runGauntlet(A);
	} catch (...) {
		*fos << "PROCESSING ERROR" << std::endl;
	}
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

void funcsBandwidth(const RCP<MAT> &A) {
	calcLowerBandwidth(A);
	calcUpperBandwidth(A);
}

void funcsDiagonalDominance(const RCP<MAT> &A) {
	calcRowDiagonalDominance(A);
	calcColDiagonalDominance(A);
}

void funcsInfNorm(const RCP<MAT> &A) {
	calcInfNorm(A);
	calcSymmetricInfNorm(A);
	calcAntisymmetricInfNorm(A);
}

void funcsNonzeros(const RCP<MAT> &A) {
	calcMinNonzerosPerRow(A);
	calcAvgNonzerosPerRow(A);
}

void funcsSum(const RCP<MAT> &A) {
	calcAbsNonzeroSum(A);
	calcNonzeroSum(A);
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
