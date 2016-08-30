#include "tpetra_properties_crsmatrix.h"

void calcDummyRows(const RCP<MAT> &A) {
	size_t rows = A->getGlobalNumRows();
	size_t locDummy = 0, result = 0;

	//  Go through each row on the current process
	for (size_t row = 0; row < rows; row++) {
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			if (A->getNumEntriesInGlobalRow(row) == 1) {
				locDummy++;
			}
		}
	}
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &locDummy, &result);
	*fos << result << SPACE;
}
