`timescale 1ns / 1ps

module ALU (
    input [2:0] op,
    input [31:0] A,
    input [31:0] B,
    output reg [31:0] res,
    output zero
    );
    
    initial begin
        res = 0;
    end
    
    always @(op or A or B) begin
        case (op)
            3'b000: res = A + B;
            3'b001: res = A - B;
            3'b010: res = B << A; // B first!
            3'b011: res = A | B;
            3'b100: res = A & B;
            3'b101: res = (A < B) ? 1 : 0;
            3'b110: res = ((A < B && A[31] == B[31]) // both pos/neg num
                         || (A[31] == 1 && B[31] == 0)) // A neg B pos
                         ? 1 : 0; // not 8'h0000001 !!!
            3'b111: res = A ^ B;
        endcase
        if (op == 3'b000 && A == 32'b1111_1111_1111_1111_1111_1111_1111_1111 && B == 1)
            res = 0;
    end

    assign zero = (res == 0) ? 1 : 0;

endmodule