#include "tpetra_properties_crsmatrix.h"

//  *fos << the maximum row locVariance for the matrix
//  The average of the squared differences from the Mean.
void calcRowVariance(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST mean, locVariance, locMaxVariance, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		comm->barrier();
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			mean = locVariance = 0.0;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
		//  Two-step approach for locVariance, could be more efficient
			for (LO col = 0; col < cols; col++) {
				mean += values[col];
			}
		//  Divide entries by the dim (to include zeros)
			mean /= A->getGlobalNumCols();
			for (LO col = 0; col < cols; col++) {
				locVariance += (values[col] - mean) * (values[col] - mean);
			}
			for (LO col = cols; col < A->getGlobalNumCols(); col++) {
				locVariance += (-mean) * (-mean);
			}
			locVariance /= A->getGlobalNumCols();
			if (locVariance > locMaxVariance) {
				locMaxVariance = locVariance;
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &locMaxVariance, &result);
	*fos << result << CSV;
}
void calcRowVariance(const RCP<MAT> &A, json &j) {
	GO rows = A->getGlobalNumRows();
	ST mean, locVariance, locMaxVariance, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		comm->barrier();
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			mean = locVariance = 0.0;
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
		//  Two-step approach for locVariance, could be more efficient
			for (LO col = 0; col < cols; col++) {
				mean += values[col];
			}
		//  Divide entries by the dim (to include zeros)
			mean /= A->getGlobalNumCols();
			for (LO col = 0; col < cols; col++) {
				locVariance += (values[col] - mean) * (values[col] - mean);
			}
			for (LO col = cols; col < A->getGlobalNumCols(); col++) {
				locVariance += (-mean) * (-mean);
			}
			locVariance /= A->getGlobalNumCols();
			if (locVariance > locMaxVariance) {
				locMaxVariance = locVariance;
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &locMaxVariance, &result);
	j["row_variance"] = result;
}
