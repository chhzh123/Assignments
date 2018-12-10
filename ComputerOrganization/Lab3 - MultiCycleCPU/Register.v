`timescale 1ns / 1ps

module Register (
    input clk,
    input reset,
    input [31:0] in,
    output [31:0] out
    );
    
    reg [31:0] data;

    // initialization
    initial data = 0;

    // read data
    assign out = data;

    // write data
    always @(negedge clk) begin
        if (reset == 0)
            data <= 0;
        else
            data <= in;
    end

endmodule