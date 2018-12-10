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

module MUX5b (
    input [1:0] Sel,
    input [4:0] A,
    input [4:0] B,
    input [4:0] C,
    output reg [4:0] res
    );
    
    always @(Sel or A or B or C) begin
        case (Sel)
            2'b00: res <= A;
            2'b01: res <= B;
            2'b10: res <= C;
        endcase
    end

endmodule

module MUX4 (
    input [1:0] Sel,
    input [31:0] A,
    input [31:0] B,
    input [31:0] C,
    input [31:0] D,
    output reg [31:0] res
    );
    
    always @(Sel or A or B or C or D) begin
        case (Sel)
            2'b00: res <= A;
            2'b01: res <= B;
            2'b10: res <= C;
            2'b11: res <= D;
        endcase
    end

endmodule