#include "tpetra_properties_crsmatrix.h"

//  The variance of the diagonal (real)
void calcDiagVariance(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST locMean = 0.0;
	ST mean = 0.0, locVariance = 0.0, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
	 		size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
					locMean += values[col];
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &locMean, &mean);
	mean /= A->getGlobalNumRows();
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			bool dne = true;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
					locVariance += (values[col] - mean) * (values[col] - mean);
					dne = false;
				}
			}
			if (dne) {
				locVariance += (-mean) * (-mean);
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &locVariance, &result);
	result /= A->getGlobalNumRows();
	*fos << result << CSV;
}
void calcDiagVariance(const RCP<MAT> &A, json &j) {
	GO rows = A->getGlobalNumRows();
	ST locMean = 0.0;
	ST mean = 0.0, locVariance = 0.0, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
	 		size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
					locMean += values[col];
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &locMean, &mean);
	mean /= A->getGlobalNumRows();
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			bool dne = true;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
					locVariance += (values[col] - mean) * (values[col] - mean);
					dne = false;
				}
			}
			if (dne) {
				locVariance += (-mean) * (-mean);
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &locVariance, &result);
	result /= A->getGlobalNumRows();
	j["diagonal_variance"] = result;
}
