`timescale 1ns / 1ps

module ShiftLeft (
    input [31:0] dataIn,
    output [31:0] dataOut
    );
    
    assign dataOut = dataIn << 2;

endmodule