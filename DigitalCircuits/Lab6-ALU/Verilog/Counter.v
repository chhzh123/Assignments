`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: SYSU
// Engineer: Chen Hongzheng
// 
// Create Date: 2018/05/16 23:28:13
// Design Name: Simple_LU
// Module Name: Counter
// Project Name: Simple_LU
// Target Devices: Xilinx Basys3
// Tool Versions: Vivado 2015.2
// Description: counter
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

// 100MHz -> 1Hz
module Counter(
    input wire clr, // clear, say reset
    input wire clk, // clock
    output reg [3:0] out // 4 bits output
    );

    // time counter
    localparam MAX_COUNT = 50_000_000; // 0.5s
    reg [25:0] count; // 26 bits to store count: 2^26 > 5*10^7
    always @ (posedge clk or posedge clr)
    begin
        if (clr == 1) // reset
            count <= 0;
        else if (count == MAX_COUNT - 1) // return 0
            count <= 0;
        else
            count <= count + 1;
    end

    // frequency divisor (flip-flop)
    reg clk_div;
    always @ (posedge clk, posedge clr)
    begin
        if (clr == 1)
            clk_div <= 0;
        else if (count == MAX_COUNT - 1) // reset
            clk_div <= ~clk_div;
        else // set
            clk_div <= clk_div;
    end

    // hex counter
    always @ (posedge clk_div, posedge clr)
    begin
        if (clr == 1)
            out <= 0;
        else if (clk_div == 1)
            out <= out + 1;
        else
            out <= out;
    end

endmodule
