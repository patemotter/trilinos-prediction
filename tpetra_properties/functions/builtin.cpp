#include "tpetra_properties_crsmatrix.h"

//  Dimension of the square matrix
void calcDim(const RCP<MAT> &A) {
	*fos << A->getGlobalNumRows() << CSV;
}
void calcDim(const RCP<MAT> &A, json &j) {
	j["dimension"] = A->getGlobalNumRows();
}

//  Frobenius norm of matrix
void calcFrobeniusNorm(const RCP<MAT> &A) {
	*fos << A->getFrobeniusNorm() << CSV;
}
void calcFrobeniusNorm(const RCP<MAT> &A, json &j) {
	j["frobenius_norm"] = A->getFrobeniusNorm();
}

//  Symmetric A_s = (A+A')/2
void calcSymmetricFrobeniusNorm(const RCP<MAT> &A){
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	*fos << A_s->getFrobeniusNorm() << CSV;
}
void calcSymmetricFrobeniusNorm(const RCP<MAT> &A, json &j){
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	j["symmetric_frobenius_norm"] = A_s->getFrobeniusNorm();
}

//  Antisymmetric A_a = (A-A')/2
void calcAntisymmetricFrobeniusNorm(const RCP<MAT> &A){
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	*fos << A_a->getFrobeniusNorm() << CSV;
}
void calcAntisymmetricFrobeniusNorm(const RCP<MAT> &A, json &j){
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	j["antisymmetric_frobenius_norm"] = A_a->getFrobeniusNorm();
}

//  Total number of nonzeros in matrix
void calcNonzeros(const RCP<MAT> &A) {
	*fos << A->getGlobalNumEntries() << CSV;
}
void calcNonzeros(const RCP<MAT> &A, json &j) {
	j["nnz"] = A->getGlobalNumEntries();
}

//  Self explanatory
void calcMaxNonzerosPerRow(const RCP<MAT> &A) {
	*fos << A->getGlobalMaxNumRowEntries() << CSV;
}
void calcMaxNonzerosPerRow(const RCP<MAT> &A, json &j) {
	j["max_nonzeros_per_row"] = A->getGlobalMaxNumRowEntries();
}

//  Number of nonzeros on diagonal
void calcDiagonalNonzeros(const RCP<MAT> &A) {
	*fos << A->getGlobalNumDiags() << CSV;
}
void calcDiagonalNonzeros(const RCP<MAT> &A, json &j) {
	j["diagonal_nnz"] = A->getGlobalNumDiags();
}
