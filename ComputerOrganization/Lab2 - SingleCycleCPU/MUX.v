`timescale 1ns / 1ps

module MUX (
    input Sel,
    input [31:0] A,
    input [31:0] B,
    output reg [31:0] res
    );
    
    always @(Sel or A or B) begin
        res <= (Sel == 0) ? A : B;
    end

endmodule

module MUX5 (
    input Sel,
    input [4:0] A,
    input [4:0] B,
    output reg [4:0] res
    );
    
    always @(Sel or A or B) begin
        res <= (Sel == 0) ? A : B;
    end

endmodule