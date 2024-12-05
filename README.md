# Random_Forest_With_Parallelism_Cuda_OpenMP_MPI
Implementation of a Random Forest Algorithm based on CART, optimized with OpenMP and CUDA for gpu parallel computing

# Execution options

Execução para sequencial:
|           Comando de execução            | descrição do que será feito |
|------------------------------------------|-----------------------------|
|                  make                    | compila o programa com g++  |
| make exec CSV=caminho/para/o/arquivo/csv |     executa o programa      |

-> ex: make && make exec CSV=datasets/iris.csv

execução para Paralelo com OPENMP (a quantidade de threads deve ser alterada manualmente no arquivo RandomForest.cpp):
|             Comando de execução          |            descrição do que será feito           |
|------------------------------------------|--------------------------------------------------|
|                   make omp               | compila o programa com g++ e com a flag -fopenmp |
| make exec CSV=caminho/para/o/arquivo/csv |                executa o programa                |

-> ex: make omp && make exec CSV=datasets/iris.csv

execução para Paralelo com CUDA (a quantidade de threads por block deve ser alterada manualmente no arquivo tree.cu):

|              Comando de execução              |             descrição do que será feito          |
|-----------------------------------------------|--------------------------------------------------|
|                  make cuda                    |            compila o programa com nvcc           |
| make exec_cuda CSV=caminho/para/o/arquivo/csv |                 executa o programa               |

-> ex: make cuda && make exec_cuda CSV=datasets/iris.csv
