`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Chen Hongzheng
// 
// Create Date: 2018/05/15 20:39:08
// Design Name: Simple_LU
// Module Name: LU
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


module LU (
    input wire [1:0] op,
    input wire A,
    input wire B,
    output reg res
    );
    
    always @(*) begin
    case (op)
        2'b00: res <= A & B;
        2'b01: res <= A | B;
        2'b10: res <= A^B;
        2'b11: res <= ~A;
    endcase
    end

endmodule