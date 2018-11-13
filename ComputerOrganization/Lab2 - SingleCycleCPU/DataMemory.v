`timescale 1ns / 1ps

module DataMemory (
    input clk,
    input [31:0] address,
    input MemWrite,
    input [31:0] dataIn, // write data
    output [31:0] dataOut // read data
    );
    
    reg [7:0] memory [0:255];

    // initialization
    integer i;
    initial begin
        for (i = 0; i < 7; i = i + 1)
            memory[i] = 0;
    end

    // read data
    assign dataOut = {memory[address + 3], memory[address + 2], memory[address + 1], memory[address]};

    // write data
    always @(posedge clk) begin
        if (MemWrite == 1 && address >= 1 && address <= 255) begin
            // little endian
            memory[address + 3] <= dataIn[31:24];
            memory[address + 2] <= dataIn[23:16];
            memory[address + 1] <= dataIn[15:8];
            memory[address] <= dataIn[7:0];
        end
    end

endmodule