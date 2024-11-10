# Random_Forest_With_Parallelism_OpenMP_MPI
Implementation of a Random Forest Algorithm based on CART, optimized with OpenMP and MPI for parallel and distributed computing

[Vídeo da apresentação](https://youtu.be/BZNAmPnop78)

# Execution options

Execução para sequencial:
|           Comando de execução            | descrição do que será feito |
|------------------------------------------|-----------------------------|
|                  make                    | compila o programa com g++  |
| make exec CSV=caminho/para/o/arquivo/csv |     executa o programa      |


-> ex: make && make exec CSV=datasets/iris.csv

execução para Paralelo com OPENMP(a quantidade de threads deve ser trocando manualmente na linha de código x da classe Cart do arquivo cart.cpp):
|             Comando de execução          |            descrição do que será feito           |
|------------------------------------------|--------------------------------------------------|
|                   make omp               | compila o programa com g++ e com a flag -fopenmp |
| make exec CSV=caminho/para/o/arquivo/csv |                executa o programa                |

-> ex: make omp && make exec CSV=datasets/iris.csv

execução para Paralelo com OPENMP(a quantidade de threads deve ser trocando manualmente na linha de código x da classe Cart do arquivo cart.cpp) e MPI:

|                                Comando de execução                               |             descrição do que será feito          |
|----------------------------------------------------------------------------------|--------------------------------------------------|
|                                      make mpi                                    | compila o programa com mpic++ e a flag -fopenmp  |
| make exec CSV=caminho/para/o/arquivo/csv NP = qunatidade_De_Processos_Para_o_MPI |      executa o programa para MPI com mpirun      |

-> ex: make mpi && make exec_mpi CSV=datasets/iris.csv NP=4
