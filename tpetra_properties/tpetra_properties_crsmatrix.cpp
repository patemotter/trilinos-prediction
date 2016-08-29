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
		RCP<MATC> A = ReaderC::readSparseFile(origFilename, comm, node, true);
	  Tpetra::RowMatrixTransposer<STC, LO, GO, NT> transposer(A);
		RCP<MATC> B = transposer.createTranspose();
		*fos << "complex" << std::endl;
		*fos << "Matrix: " << filename << std::endl;
  	*fos << "Procs: " << comm->getSize() << std::endl;
	  runGauntlet(A);
	} else {
		RCP<MAT> A = Reader::readSparseFile(origFilename, comm, node, true);
	  Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer(A);
		RCP<MAT> B = transposer.createTranspose();
		*fos <<  filename << ", ";
	  runGauntlet(A);
	}
}

void runGauntlet(const RCP<MAT> &A) {
	// Test squareness
	if (A->getGlobalNumRows() != A->getGlobalNumCols() ) {
		*fos << "Not a square matrix, exiting." << std::endl;
		exit(-1);
	}
	*fos << calcRowVariance(A) << ", ";
	*fos << calcColVariance(A) << ", ";
	*fos << calcDiagVariance(A) << ", ";
	*fos << calcNonzeros(A) << ", ";
	*fos << calcDim(A) << ", ";
	*fos << calcFrobeniusNorm(A) << ", ";
	*fos << calcSymmetricFrobeniusNorm(A) << ", ";
	*fos << calcAntisymmetricFrobeniusNorm(A) << ", ";
	*fos << calcOneNorm(A) << ", ";
	*fos << calcInfNorm(A) << ", ";
	*fos << calcSymmetricInfNorm(A) << ", ";
	*fos << calcAntisymmetricInfNorm(A) << ", ";
	*fos << calcMaxNonzerosPerRow(A) << ", ";
	*fos << calcMinNonzerosPerRow(A) << ", ";
	*fos << calcAvgNonzerosPerRow(A) << ", ";
	*fos << calcTrace(A) << ", ";
	*fos << calcAbsTrace(A) << ", ";
	*fos << calcDummyRows(A) << ", ";
	calcSymmetry(A);
	*fos << calcRowDiagonalDominance(A) << ", ";
	*fos << calcColDiagonalDominance(A) << ", ";
	*fos << calcLowerBandwidth(A) << ", ";
	*fos << calcUpperBandwidth(A) << ", ";
	*fos << calcDiagonalMean(A) << ", ";
	*fos << calcDiagonalSign(A) << ", ";
	*fos << calcDiagonalNonzeros(A) << ", ";
	*fos << calcAbsNonzeroSum(A) << std::endl;
  //Values(A, "LM");
  //calcEigenValues(A, "LR");
  //calcEigenValues(A, "SM");
  //calcEigenValues(A, "SR");gT
}
void runGauntlet(const RCP<MATC> &A) {
	// Test squareness
	if (A->getGlobalNumRows() != A->getGlobalNumCols() ) {
		*fos << "Not a square matrix, exiting." << std::endl;
		exit(-1);
	}
	*fos << "Complex!, ";
	/*
	*fos << comm->getSize() << ", ";
	*fos << calcRowVariance(A) << ", ";
	*fos << calcColVariance(A) << ", ";
	*fos << calcDiagVariance(A) << ", ";
	*fos << calcNonzeros(A) << ", ";
	*fos << calcDim(A) << ", ";
	*fos << calcFrobeniusNorm(A) << ", ";
	*fos << calcSymmetricFrobeniusNorm(A) << ", ";
	*fos << calcAntisymmetricFrobeniusNorm(A) << ", ";
	*fos << calcOneNorm(A) << ", ";
	*fos << calcInfNorm(A) << ", ";
	*fos << calcSymmetricInfNorm(A) << ", ";
	*fos << calcAntisymmetricInfNorm(A) << ", ";
	*fos << calcMaxNonzerosPerRow(A) << ", ";
	*fos << calcMinNonzerosPerRow(A) << ", ";
	*fos << calcAvgNonzerosPerRow(A) << ", ";
	*fos << calcTrace(A) << ", ";
	*fos << calcAbsTrace(A) << ", ";
	*fos << calcDummyRows(A) << ", ";
	calcSymmetry(A);
	*fos << calcRowDiagonalDominance(A) << ", ";
	*fos << calcColDiagonalDominance(A) << ", ";
	*fos << calcLowerBandwidth(A) << ", ";
	*fos << calcUpperBandwidth(A) << ", ";
	*fos << calcDiagonalMean(A) << ", ";
	*fos << calcDiagonalSign(A) << ", ";
	*fos << calcDiagonalNonzeros(A) << ", ";
  calcEigenValues(A, "LM");
  calcEigenValues(A, "LR");
  calcEigenValues(A, "SM");
  calcEigenValues(A, "SR");
  */
  *fos << std::endl;
}
