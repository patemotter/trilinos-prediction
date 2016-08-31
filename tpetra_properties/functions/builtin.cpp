#include "tpetra_properties_crsmatrix.h"

//  Dimension of the square matrix
void calcDim(const RCP<MAT> &A) {
	*fos << A->getGlobalNumRows() << CSV;
}

//  Frobenius norm of matrix
void calcFrobeniusNorm(const RCP<MAT> &A) {
	*fos << A->getFrobeniusNorm() << CSV;
}

//  Symmetric A_s = (A+A')/2
void calcSymmetricFrobeniusNorm(const RCP<MAT> &A){
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	*fos << A_s->getFrobeniusNorm() << CSV;
}

//  Antisymmetric A_a = (A-A')/2
void calcAntisymmetricFrobeniusNorm(const RCP<MAT> &A){
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	*fos << A_a->getFrobeniusNorm() << CSV;
}

//  Total number of nonzeros in matrix
void calcNonzeros(const RCP<MAT> &A) {
	*fos << A->getGlobalNumEntries() << CSV;
}

//  Self explanatory
void calcMaxNonzerosPerRow(const RCP<MAT> &A) {
	*fos << A->getGlobalMaxNumRowEntries() << CSV;
}

void calcDiagonalNonzeros(const RCP<MAT> &A) {
	*fos << A->getGlobalNumDiags() << CSV;
}
