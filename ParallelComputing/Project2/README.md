## 并行与分布式计算 - 最短路径

* 编译运行方法如下，编译变量均已在`Makefile`中声明

```bash
# OpenMP
$ make
$ make run

# OpenMP + MPI
$ make MPI=1
$ make run_mpi
```

部分编译变量声明
* `TEST`：需要测试的数据集路径，默认为`10001x10001GraphExamples.txt`
* `NUM_VERT`：测试集的顶点数目，默认为`20001`
* `NUM_CORE`：进程数目，默认为`3`

注意，在单机上测试很可能OpenMP执行的效果要远胜于使用OpenMP和MPI的效果。