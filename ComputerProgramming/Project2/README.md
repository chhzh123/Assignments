# 多项式计算器

## 简介
本多项式计算器由陈鸿峥用`C++`开发，用`g++ (MinGW.org GCC-6.3.0-1) 6.3.0`进行编译。

## 运行环境
* `Windows`系统下直接打开`polynomial_calculator.exe`文件即可，若出现缺`dll`文件的情况，请自行重新编译（因`gcc`默认进行动态链接编译，故放至其他电脑经常会出现这种情况）
* 如要自行编译，请确保电脑的`C++`编译器支持`C++11`标准，并且要将`polynomial_calculator.cpp`、`polynomials.hpp`、`polynomial.hpp`放在同一文件夹下
* 文件运行时会产生`PolynomialsData.dat`文件，请千万不要删除，里面是多项式数据的存档。生成后也请不要移动，保持该文件与`exe`文件处在同个文件夹下。
* 如要修改`PolynomialsData.dat`文件，请保持格式不变，并不要添加多余符号及换行符。