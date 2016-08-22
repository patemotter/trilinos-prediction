#include "tpetra_properties_crsmatrix.h"

//  Dimension of the square matrix
size_t calcDim(const RCP<MAT> &A) {
	TimeMonitor LocalTimer (*timeDim);
	return A->getGlobalNumRows();
}
size_t calcDim(const RCP<MATC> &A) {
	TimeMonitor LocalTimer (*timeDim);
	return A->getGlobalNumRows();
}

//  Frobenius norm of matrix
ST calcFrobeniusNorm(const RCP<MAT> &A) {
	TimeMonitor LocalTimer (*timeFrobeniusNorm);
	return A->getFrobeniusNorm();
}
ST calcFrobeniusNorm(const RCP<MATC> &A) {
	TimeMonitor LocalTimer (*timeFrobeniusNorm);
	return A->getFrobeniusNorm();
}

//  Symmetric A_s = (A+A')/2
ST calcSymmetricFrobeniusNorm(const RCP<MAT> &A){
	TimeMonitor LocalTimer (*timeSymmetricFrobeniusNorm);
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	return A_s->getFrobeniusNorm();
}

//  Antisymmetric A_a = (A-A')/2
ST calcAntisymmetricFrobeniusNorm(const RCP<MAT> &A){
	TimeMonitor LocalTimer (*timeAntisymmetricFrobeniusNorm);
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	return A_a->getFrobeniusNorm();
}

//  Total number of nonzeros in matrix
size_t calcNonzeros(const RCP<MAT> &A) {
	TimeMonitor LocalTimer (*timeNonzeros);
	return A->getGlobalNumEntries();
}

//  Self explanatory
size_t calcMaxNonzerosPerRow(const RCP<MAT> &A) {
	TimeMonitor LocalTimer (*timeMaxNonzerosPerRow);
	return A->getGlobalMaxNumRowEntries();
}

size_t calcDiagonalNonzeros(const RCP<MAT> &A) {
	TimeMonitor LocalTimer (*timeDiagonalNonzeros);
	return A->getGlobalNumDiags();
}
