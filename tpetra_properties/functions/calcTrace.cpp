#include "tpetra_properties_crsmatrix.h"

void calcTrace(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST trace = 0.0, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
          trace += values[col];
        }
      }
    }
  }
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &trace, &result);
  *fos << result << SPACE;
}

void calcAbsTrace(const RCP<MAT> &A) {
	GO rows = A->getGlobalNumRows();
	ST trace = 0.0, result = 0.0;

	//  Go through each row on the current process
	for (GO row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			size_t cols = A->getNumEntriesInGlobalRow(row);
			Array<ST> values(cols);
			Array<GO> indices(cols);
			A->getGlobalRowCopy(row, indices(), values(), cols);
			for (size_t col = 0; col < cols; col++) {
				if (indices[col] == row) {
					trace += fabs(values[col]);
				}
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &trace, &result);
	*fos << result << SPACE;
}
