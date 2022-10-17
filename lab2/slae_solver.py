from os import stat
import numpy as np
EPSILON = 1E-15

class SlaeSolver :

    #Прямой ход
    def _forward_elimination(A, F): 
        n = len(A)

        for i in range(0, n):
            #выбираем элемент
            for j in range(i + 1, n):
                if A[j][i] > A[i][i]:
                    A[i], A[j] = A[j], A[i]
                    F[i], F[j] = F[j], F[i]
            
            for j in range(i + 1, n):
                F[j] -= F[i] * (A[j][i] / A[i][i])
                A[j] -= A[i] * (A[j][i] / A[i][i])

            F[i] /= A[i][i]
            A[i] /= A[i][i]

        return A, F    

    #обратный ход
    def _back_substitution(A, F):
        n = len(A)
        sum = 0

        solution = [0 for _ in range(n)]
        
        # хоть и после _forward_elimination диагональные элементы нормированы,
        # все равно делаем это заново, вдруг метод будет вызван не из gauss_solve.
        solution[n-1] = F[n-1] / A[n-1][n-1]

        for i in range(n-2, -1, -1):
            solution[i] = 1 / A[i][i] * (F[i] - np.matmul(A[i], solution))

        return solution


    @staticmethod
    def gauss_solve(A, F):
        copyA = A.copy()
        copyF = F.copy()
        
        copyA, copyF = SlaeSolver._forward_elimination(copyA, copyF)

        return SlaeSolver._back_substitution(copyA, copyF)


    def _LU_decompose(A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))

        for i in range(0, n):
            L[i][i] = 1

        for i in range(0, n):
            for j in range(0, n):
                sum = 0
                if i <= j:
                    for k in range(0, i):
                        sum += L[i][k] * U[k][j]
                    U[i][j] = A[i][j] - sum
                if i > j:
                    for k in range(0, j):
                        sum += L[i][k] * U[k][j]
                    L[i][j] = (A[i][j] - sum) / U[j][j]

        return L, U

    @staticmethod
    def LU_solve(A, F):
        copyF = F.copy()

        L, U = SlaeSolver._LU_decompose(A)

        SlaeSolver._forward_elimination(L, copyF)
        return SlaeSolver._back_substitution(U, copyF)


    @staticmethod
    def upper_relaxation_solve(A, F, param):
        n = len(A)
        copyA = A.copy()
        copyF = F.copy()

        L = np.zeros((n, n))
        D = np.zeros((n, n))
        U = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    D[i][j] = A[i][j]
                elif i > j:
                    L[i][j] = A[i][j]
                else:
                    U[i][j] = A[i][j]

        inv = np.linalg.inv(D + param * L)
        B = - np.matmul(inv, (param - 1) * D + param * U)
        f = np.matmul(param * inv, F)

        x = np.array([0.5 for i in range (0, n)])

    
        while np.linalg.norm(copyF - np.matmul(copyA, x)) > EPSILON:
            x = np.matmul(B, x) + f


        return x

