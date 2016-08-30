#include "tpetra_properties_crsmatrix.h"

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
	bool complex = false;

	//  Decide to print to screen or file
	if (outputDir.empty()) {
		//std::cout << "No output directory was specified. Printing to screen" << std::endl;
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
			complex = true;
		}
	}
	infile.close();
	if (complex) {
		*fos << "Complex matrices are not currently supported: exiting\n";
		exit(-1);
	} else {
		RCP<MAT> A = Reader::readSparseFile(origFilename, comm, node, true);
		Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer(A);
		RCP<MAT> B = transposer.createTranspose();
		*fos <<  filename;
		runGauntlet(A);
	}
}

void runGauntlet(const RCP<MAT> &A) {
	// Test squareness
	if (A->getGlobalNumRows() != A->getGlobalNumCols() ) {
		*fos << "Not a square matrix, exiting." << std::endl;
		exit(-1);
	}
	calcRowVariance(A);
	calcColVariance(A);
	calcDiagVariance(A);
	calcNonzeros(A);
	calcDim(A);
	calcFrobeniusNorm(A);
	calcSymmetricFrobeniusNorm(A);
	calcAntisymmetricFrobeniusNorm(A);
	calcOneNorm(A);
	calcInfNorm(A);
	calcSymmetricInfNorm(A);
	calcAntisymmetricInfNorm(A);
	calcMaxNonzerosPerRow(A);
	calcMinNonzerosPerRow(A);
	calcAvgNonzerosPerRow(A);
	calcTrace(A);
	calcAbsTrace(A);
	calcDummyRows(A);
	calcSymmetry(A);
	calcRowDiagonalDominance(A);
	calcColDiagonalDominance(A);
	calcLowerBandwidth(A);
	calcUpperBandwidth(A);
	calcDiagonalMean(A);
	calcDiagonalSign(A);
	calcDiagonalNonzeros(A);
	calcAbsNonzeroSum(A);
}
