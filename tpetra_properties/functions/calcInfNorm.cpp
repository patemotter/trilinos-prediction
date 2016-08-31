#include "tpetra_properties_crsmatrix.h"

//  Max absolute row sum
void calcInfNorm(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST locSum, locMaxSum, result = 0.0;
	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			locSum = 0;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (LO col = 0; col < cols; col++) {
				locSum += fabs(values[col]);
			}
			if (locSum > locMaxSum) {
				locMaxSum = locSum;
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &locMaxSum, &result);
	*fos << result << CSV;
}
void calcInfNorm(const RCP<MAT> &A, json &j, const int &flag) {
	GO rows = A->getGlobalNumRows();
	ST locSum, locMaxSum, result = 0.0;
	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			locSum = 0;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (LO col = 0; col < cols; col++) {
				locSum += fabs(values[col]);
			}
			if (locSum > locMaxSum) {
				locMaxSum = locSum;
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &locMaxSum, &result);
	if (flag == 0) {
		j["infinity_norm"] = result;
	} else if (flag == 1) {
		j["symmetric_infinity_norm"] = result;
	} else if (flag == 2) {
		j["symmetric_infinity_norm"] = result;
	}
}

//  Max absolute row sum of symmetric part
void calcSymmetricInfNorm(const RCP<MAT> &A) {
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	calcInfNorm(A_s);
}
void calcSymmetricInfNorm(const RCP<MAT> &A, json &j) {
	RCP<MAT> A_s = Tpetra::MatrixMatrix::add(0.5, false, *A, 0.5, true, *A);
	calcInfNorm(A_s, j, 1);
}

//  Max absolute row sum of anti-symmetric part
void calcAntisymmetricInfNorm(const RCP<MAT> &A) {
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	calcInfNorm(A_a);
}
void calcAntisymmetricInfNorm(const RCP<MAT> &A, json &j) {
	RCP<MAT> A_a = Tpetra::MatrixMatrix::add(0.5, false, *A, -0.5, true, *A);
	calcInfNorm(A_a, j, 2);
}
