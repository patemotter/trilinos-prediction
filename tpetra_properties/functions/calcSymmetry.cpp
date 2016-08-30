#include "tpetra_properties_crsmatrix.h"

/*
*	Determines the symmetry of a square matrix based on non-diagonal nonzeros
*	- 'match' exact numeric matches between two nonzero entries
*	- 'noMatch' numeric disagreements between two nonzero entries
*	- 'dne' nonzero entries which match to a zero entry
*/
void calcSymmetry(const RCP<MAT> &A) {
	//  A is the original matrix, B is its transpose
	Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer(A);
	RCP<MAT> B = transposer.createTranspose();

	GO rows = A->getGlobalNumRows();
	ST result = 0.0;
	GO match = 0, noMatch = 0, dne = 0;
	GO totalMatch, totalNoMatch, totalDne;
	GO locEntries = 0;

	GO diagNonzeros = A->getGlobalNumDiags();
	GO offDiagNonzeros = A->getGlobalNumEntries() - diagNonzeros;
	for (GO row = 0; row < rows; row++) {
		//  Limit the work to whichever node hosts the data
		if (A->getRowMap()->isNodeGlobalElement(row)) {
			size_t colsA = A->getNumEntriesInGlobalRow(row);
			size_t colsB = B->getNumEntriesInGlobalRow(row);
			Array<ST> valuesA(colsA), valuesB(colsB);
			Array<GO> indicesA(colsA), indicesB(colsB);
			A->getGlobalRowCopy(row, indicesA(), valuesA(), colsA);
			B->getGlobalRowCopy(row, indicesB(), valuesB(), colsB);

			//  Make maps for each row, ignoring diagonal
			std::map<GO, ST> mapA, mapB;
			for (int colA = 0; colA < colsA; colA++) {
				if (row != indicesA[colA])
					mapA.insert( std::pair<GO,ST>(indicesA[colA], valuesA[colA]) );
			}
			for (int colB = 0; colB < colsB; colB++) {
				if (row != indicesB[colB])
					mapB.insert( std::pair<GO,ST>(indicesB[colB], valuesB[colB]) );
			}
			//  Compare the ma
			std::map<GO, ST>::iterator iterA;
			for (iterA = mapA.begin(); iterA != mapA.end(); iterA++) {
				//  Matching indices found
				if (mapB.count (iterA->first) ) {
					//  Check if values for those indices match
					if ( iterA->second == mapB[iterA->first] ) {
						match++;
					} else {
						noMatch++;
					}
				} else {
					dne++;
				}
			}
		}
	}
	//  Gather all results and compute percentages
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &match, &totalMatch);
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &noMatch, &totalNoMatch);
	Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &dne, &totalDne);
	*fos << (double)totalMatch/(double)offDiagNonzeros << ", ";
	*fos << (double)totalNoMatch/(double)offDiagNonzeros << ", ";
	*fos << (double)totalDne/(double)offDiagNonzeros << ", ";
	totalMatch == offDiagNonzeros ? *fos << 1 : *fos << 0; *fos << SPACE;
	totalNoMatch == offDiagNonzeros ? *fos << 1 : *fos << 0; *fos << SPACE;
	totalDne == offDiagNonzeros ? *fos << 1 : *fos << 0; *fos << SPACE;
}
