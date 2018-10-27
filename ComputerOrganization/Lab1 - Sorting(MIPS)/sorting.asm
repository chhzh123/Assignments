# Copyright 2018, SYSU
# Author: Chen Hongzheng, Major in Computer Science (17), SDCS, SYSU
# E-mail: chenhzh37@mail2.sysu.edu.cn
# Date: 2018/10/09
# This is a simple sorting (selection, inverse ordering) program for 10 numbers.

.text
.globl main

# ------------ register description -------------
# $s7=10, $s6=9              loop boundary (constant)
# $t0(out), $t1(in)          array address
# $t7(out), $t6(in)          loop counter
# $t9(out), $t8(in)          bool value used for loop termination judgement
# $s1(curr), $s2(global)     temporary value used for find the maximum
# $s0=arr[$t0], $s1=arr[$t1]
# $t2                        maximum address
# $v0                        system function call (input/output)

# ----------- main function segment -------------
main:
# -------------------- input --------------------
    addi    $s7,    $zero,    10       # loop boundary
    add     $t7,    $zero,    $zero    # counter
    la      $t0,    array              # address of the array
input_loop:
    addi    $t7,    $t7  ,    1        # $t7<-$t7+1, start from 1
    li      $v0,    5                  # load integer
    syscall
    addi    $t0,    $t0  ,    4        # calculate the address of arr[$t0], start from arr[1]
    sw      $v0,    0($t0)             # store the input number into arr[$t0]

    slt     $t9,    $t7  ,    $s7      # $t7<10 ? $t9=1 : $t9=0
    bnez    $t9,    input_loop         # $t9!=0  ? loop  : continue

# ------------------- sorting -------------------
    addi    $s7,    $zero,    10       # loop boundary
    addi    $s6,    $zero,    9        # outer loop boundary, NOTE THAT NOT 10!
    add     $t7,    $zero,    $zero    # outer_counter
    la      $t0,    array              # address of the array
outer_loop:
    addi    $t7,    $t7  ,    1        # $t7<-$t7+1, start from 1
    addi    $t0,    $t0  ,    4        # calculate the address of arr[$t0], start from arr[1]
    lw      $s0,    0($t0)             # load arr[$t0] from the memory

    add     $t6,    $t7  ,    $zero    # inner_counter initialization, $t6=$t7
    add     $t1,    $t0  ,    $zero    # initialize the address, $t1=$t0
    lw      $s2,    0($t0)             # load arr[$t0] as the initial maximum
    addi    $t2,    $t0  ,    0        # initial maximum address, $t0

    # !!! since there're 10 numbers, inner loop must be executed 
    inner_loop:
        addi $t6,    $t6,     1        # $t6<-$t6+1, start from $t7+1
        addi $t1,    $t1,     4        # calculate the address of arr[$t1], start from arr[$t0+1]
        lw   $s1,    0($t1)            # load arr[$t1] from the memory

        slt  $t8,    $s1,     $s2      # arr[$t0]<$s2, do nothing
        bnez $t8,    no_update         # update or not
        add  $s2,    $s1,     $zero    # update the maximum
        add  $t2,    $t1,     $zero    # update the maximum address

        no_update:
        slt  $t8,    $t6,     $s7      # $t6<10 ? continue : break
        bnez $t8,    inner_loop
    
    # swap two numbers, $s0 has been the temporary variable
    sw      $s2,     0($t0)            # store the maximum into arr[$t0]
    sw      $s0,     0($t2)            # store original arr[$t0] into arr[$t2]

    # loop condition
    slt     $t9,     $t7,     $s6      # $t7<9 ? continue : break
    bnez    $t9,     outer_loop

# -------------------- output -------------------
    addi    $s7,    $zero,    10       # loop boundary
    add     $t7,    $zero,    $zero    # counter
    la      $t0,    array              # address of the array
output_loop:
    addi    $t7,    $t7  ,    1        # $t7<-$t7+1, start from 1
    addi    $t0,    $t0  ,    4        # calculate the address of arr[$t0], start from arr[1]
    lw      $a0,    0($t0)             # load arr[$t0]
    li      $v0,    1                  # print integer
    syscall
    la      $a0,    nline              # prepare for output
    li      $v0,    4                  # print space
    syscall

    slt     $t9,    $t7  ,    $s7      # $t7<$s7 ? $t9=1 : $t9=0
    bnez    $t9,    output_loop        # $t9!=0  ? loop  : continue

# --------------------- exit --------------------
    li      $v0,    10                 # exit
    syscall

# ------------- data field segment --------------
.data
    array:     .space  44              # 11*4=44bytes, int for 4 bytes
    inputline: .asciiz "Please enter 10 numbers to sort: "
    nline:     .asciiz " "