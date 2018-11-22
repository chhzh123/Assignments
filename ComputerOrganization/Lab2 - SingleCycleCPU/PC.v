`timescale 1ns / 1ps

module PC (
    input clk,
    input reset,
    input PCWrite,
    input [31:0] nextPC,
    output reg [31:0] currPC
    );

    initial currPC = 0;

    always @(posedge clk or negedge reset) begin
        if (reset == 0)
            currPC <= 32'hFFFFFFFC;
        else if (PCWrite == 1)
            currPC <= nextPC;
    end

endmodule