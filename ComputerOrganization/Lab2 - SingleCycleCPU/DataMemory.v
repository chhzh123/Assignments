`timescale 1ns / 1ps

module DataMemory (
    input clk,
    input reset,
    input [31:0] address,
    input MemWrite,
    input [31:0] dataIn, // write data
    output [31:0] dataOut // read data
    );
    
    reg [7:0] memory [0:255]; // 8 bits (bandwidth) * #256 (address)

    // initialization
    integer i;
    initial begin
        for (i = 0; i < 256; i = i + 1)
            memory[i] = 0;
    end

    // read data
    assign dataOut = (address == 0) ? 0 : {memory[address + 3],
                                           memory[address + 2],
                                           memory[address + 1],
                                           memory[address]};

    // write data
    always @(negedge clk) begin
        if (reset == 0)
            memory[0] <= 0;
        else if (MemWrite == 1 && address != 0) begin // do not use <=255!!!
            // little endian
            memory[address + 3] <= dataIn[31:24];
            memory[address + 2] <= dataIn[23:16];
            memory[address + 1] <= dataIn[15:8];
            memory[address] <= dataIn[7:0];
        end
    end

endmodule