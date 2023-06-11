import numpy as np

def singularValuesDecomposition(A):
    #Niech B będzie iloczynem macierzy A i A^T,
    #a C będzie iloczyn macierzy A^T i A
    B = np.dot(A, A.T)
    C = np.dot(A.T, A)

    eigenvaluesB, eigenvectorsB = np.linalg.eig(B)
    eigenvaluesB = np.sqrt(np.abs(eigenvaluesB))
    eigenvectorsB = eigenvectorsB[:, np.argsort(-eigenvaluesB)]
    eigenvaluesB = np.sort(eigenvaluesB)[::-1]

    eigenvaluesC, eigenvectorsC = np.linalg.eig(C)
    eigenvectorsC = eigenvectorsC[:, np.argsort(-eigenvaluesC)]

    m, n = A.shape
    S = np.zeros((m, n))
    min_dim = min(m, n)
    S[:min_dim, :min_dim] = np.diag(eigenvaluesB)

    eigenvectorsB[np.abs(eigenvectorsB) < 1e-15] = 0
    eigenvectorsC[np.abs(eigenvectorsC) < 1e-15] = 0
    return eigenvectorsB, S, eigenvectorsC.T

def printDecomposition(matrix):
    U, S, V = singularValuesDecomposition(matrix)

    print(f"Singular Values Decomposition for matrix: \n{A}\n"
          f"---\nU:\n{U}\n"
          f"---\nS:\n{S}\n"
          f"---\nV:\n{V}\n")

if __name__ == '__main__':
    A = np.array([[3,2,2], [2,3,-2]])

    printDecomposition(A)
