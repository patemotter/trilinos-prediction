#include "tpetra_properties_crsmatrix.h"

// 0 not, 1 weak, 2 strict
// a_ii >= sum(a_ij) for all i,js i!=j
void calcRowDiagonalDominance(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST result = 0.0;
	GO totalMatch, match = 0;
	GO locEntries = 0;
	ST locDiagSum = 0.0, locRowSum = 0.0;
	ST totalDiagSum, totalRowSum;
	int strict = 1, totalStrict;

	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			if (cols < A->getGlobalNumRows()) {
				totalDiagSum = totalRowSum = 0.0;
				for (size_t col = 0; col < cols; col++) {
					if (row == indices[col]) {
						totalDiagSum += values[col];
					} else {
						totalRowSum += values[col];
					}
				}
				if (locDiagSum < locRowSum) {
					*fos << 0;
				} else if (locDiagSum == locRowSum) {
					strict = 0;
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &strict, &totalStrict);
	if (totalStrict == 1) {
		*fos << 2 << CSV;
	} else {
		*fos << 1 << CSV;
	}
}
void calcRowDiagonalDominance(const RCP<MAT> &A, json &j) {
	GO rows = A->getGlobalNumRows();
	ST result = 0.0;
	GO totalMatch, match = 0;
	GO locEntries = 0;
	ST locDiagSum = 0.0, locRowSum = 0.0;
	ST totalDiagSum, totalRowSum;
	int strict = 1, totalStrict;

	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			if (cols < A->getGlobalNumRows()) {
				totalDiagSum = totalRowSum = 0.0;
				for (size_t col = 0; col < cols; col++) {
					if (row == indices[col]) {
						totalDiagSum += values[col];
					} else {
						totalRowSum += values[col];
					}
				}
				if (locDiagSum < locRowSum) {
					j["row_diagonal_dominance"] = 0;
					return;
				} else if (locDiagSum == locRowSum) {
					strict = 0;
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &strict, &totalStrict);
	if (totalStrict == 1) {
		j["row_diagonal_dominance"] = 2;
	} else {
		j["row_diagonal_dominance"] = 1;
	}
}

void calcColDiagonalDominance(const RCP<MAT> &A) {
	Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer(A);
	RCP<MAT> B = transposer.createTranspose();

	GO rows = B->getGlobalNumRows();
	ST result = 0.0;
	GO totalMatch, match = 0;
	GO locEntries = 0;
	ST locDiagSum = 0.0, locRowSum = 0.0;
	ST totalDiagSum, totalRowSum;
	int strict = 1, totalStrict;

	for (GO row = 0; row < rows; row++) {
		if (B->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = B->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			B->getGlobalRowCopy(row, indices(), values(), cols);
			if (cols < B->getGlobalNumRows()) {
				totalDiagSum = totalRowSum = 0.0;
				for (size_t col = 0; col < cols; col++) {
					if (row == indices[col]) {
						totalDiagSum += values[col];
					} else {
						totalRowSum += values[col];
					}
				}
				if (locDiagSum < locRowSum) {
					*fos << 0;
				} else if (locDiagSum == locRowSum) {
					strict = 0;
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &strict, &totalStrict);
	if (totalStrict == 1) {
		*fos << 2 << CSV;
	} else {
		*fos << 1 << CSV;
	}
}
void calcColDiagonalDominance(const RCP<MAT> &A, json &j) {
	Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer(A);
	RCP<MAT> B = transposer.createTranspose();

	GO rows = B->getGlobalNumRows();
	ST result = 0.0;
	GO totalMatch, match = 0;
	GO locEntries = 0;
	ST locDiagSum = 0.0, locRowSum = 0.0;
	ST totalDiagSum, totalRowSum;
	int strict = 1, totalStrict;

	for (GO row = 0; row < rows; row++) {
		if (B->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = B->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			B->getGlobalRowCopy(row, indices(), values(), cols);
			if (cols < B->getGlobalNumRows()) {
				totalDiagSum = totalRowSum = 0.0;
				for (size_t col = 0; col < cols; col++) {
					if (row == indices[col]) {
						totalDiagSum += values[col];
					} else {
						totalRowSum += values[col];
					}
				}
				if (locDiagSum < locRowSum) {
					j["col_diagonal_dominance"] = 0;
					return;
				} else if (locDiagSum == locRowSum) {
					strict = 0;
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &strict, &totalStrict);
	if (totalStrict == 1) {
		j["col_diagonal_dominance"] = 2;
	} else {
		j["col_diagonal_dominance"] = 1;
	}
}
