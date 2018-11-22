`timescale 1ns / 1ps

module Registers (
    input clk,
    input reset,
    input [4:0] r1, // read reg #1 address
    input [4:0] r2, // read reg #2 address
    input [4:0] wr, // write reg address
    input RegWrite,
    input [31:0] wd, // write data
    output [31:0] d1, // read data 1
    output [31:0] d2 // read data 2
    );
    
    reg [31:0] register [0:31]; // 32 bits (bandwidth) * #32 (address)
    
    // initialization
    integer i;
    initial begin
        for (i = 0; i < 32; i = i + 1)
            register[i] = 0;
    end

    // read data
    assign d1 = (r1 == 0) ? 0 : register[r1];
    assign d2 = (r2 == 0) ? 0 : register[r2];

    // write data
    always @(negedge clk) begin
        if (reset == 0)
            register[0] <= 0;
        else if (wr != 0 && RegWrite == 1)
            register[wr] <= wd;
    end

endmodule