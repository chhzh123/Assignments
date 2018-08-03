`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Chen Hongzheng
// 
// Create Date: 2018/05/17
// Design Name: Simple_LU
// Module Name: Wrapper
// Project Name: Simple_LU
// Target Devices: Xilinx Basys3
// Tool Versions: Vivado 2015.2
// Description: LU
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module wrapper (
    input clk,
    output wire A,
    output wire B,
    output wire [1:0] op_code,
    output wire res
    );
    
    wire [2:0] out;
    Counter Counter(.clr(clr), .clk(clk), .out({op_code, A, B}));
    LU LU(.op(op_code), .A(A), .B(B), .res(res));

endmodule