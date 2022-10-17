from calendar import prcal
import numpy as np
import slae_solver as sv
EPSILON = 1E-15
# задаем СЛАУ (система "К")
n = 10

matrix = np.zeros((n,n))

for i in range(0, n):
    for j in range(0, n):
        if i==j:
            matrix[i][j] = 1
        else: 
            matrix[i][j] = 1 / (i + j + 2)

vector = np.array( [(1 / (1+i) ) for i in range(0, n)])

# ***********************************************************
def checkLU(matrix):
    for i in range(0, n):
        mat_i = np.array([lines[0: i + 1] for lines in matrix[0: i + 1]])
        if not np.linalg.det(mat_i):
            return False
    return True



# получиться максимальное и минимальное собственные значения
def get_eigen_values(matrix):
    return get_eigen_max(matrix), 1/get_eigen_max(np.linalg.inv(matrix))


# получить максимальное собственное значение
def get_eigen_max(matrix):
    y_prev = np.array([0.1 for i in range (0, len(matrix))])
    y_cur = np.matmul(matrix, y_prev)

    eps = 1e-6
    while (np.linalg.norm(np.matmul(matrix, y_cur))) / np.linalg.norm( y_cur) - np.linalg.norm( y_cur) / np.linalg.norm(y_prev) > eps:
        y_prev = y_cur
        y_cur = np.matmul(matrix, y_cur)

    return np.linalg.norm(np.matmul(matrix, y_cur)) / np.linalg.norm(y_cur)

# число обусловленностей
def get_condition_number(matrix):
    return np.linalg.norm(matrix) * np.linalg.norm(np.transpose(matrix))


# ***********************************************************
# main
# ***********************************************************

max, min = get_eigen_values(matrix)
print(max/min)
print("\nСобственные значения степенным методом\n")
print("lambda_max:", max)
print("lambda_min:", min)
print('\n*********************************************************************************************\n')

print("\nСобственные значения через numpy\n")
eigs, v = np.linalg.eig(matrix)
eigs.sort()
print("lambda_max:", eigs[9])
print("lambda_min:", eigs[0])
print('\n*********************************************************************************************\n')


print("\nЧисло обусловленности матрицы\n")
print(get_condition_number(matrix))
print('\n*********************************************************************************************\n')


print("Критерий останова: |Ax - f| < eps =", EPSILON)
print('\n*********************************************************************************************\n')

print("Проверка на LU")
print(checkLU(matrix))
print('\n*********************************************************************************************\n')

print("LU разложение\n")
x = sv.SlaeSolver.LU_solve(matrix, vector)
print("Корень системы:\n", x)
print("Невязка метода: ", np.linalg.norm(vector - np.matmul(matrix, x)))
print('\n*********************************************************************************************\n')

print("Метод Гаусса мой\n")
x = sv.SlaeSolver.gauss_solve(matrix, vector)
print("Корень системы:\n", x)
print("Невязка метода: ", np.linalg.norm(vector - np.matmul(matrix, x)))
print('\n*********************************************************************************************\n')

print("Метод верхней релаксации:\n")
y = sv.SlaeSolver.upper_relaxation_solve(matrix, vector, 1.5)
print("Корень системы:\n", y)
print("Невязка метода: ", np.linalg.norm(vector - np.matmul(matrix, x)))
print('\n*********************************************************************************************\n')

print("Метод из numpy\n")
x = np.linalg.solve(matrix, vector)
print("Корень системы:\n", x)
print("Невязка метода: ", np.linalg.norm(vector - np.matmul(matrix, x)))
print('\n*********************************************************************************************\n')