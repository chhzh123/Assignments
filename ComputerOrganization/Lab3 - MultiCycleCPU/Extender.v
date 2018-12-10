`timescale 1ns / 1ps

module Extender (
	input Sel,
	input [15:0] dataIn,
	output reg [31:0] dataOut
	);

	initial dataOut = 0;

	always @(Sel or dataIn) begin // dataIn!!!
		if (Sel == 0) // ZeroExt
			dataOut = {{16{1'b0}},dataIn[15:0]};
		else // SignExt
			dataOut = {{16{dataIn[15]}},dataIn[15:0]};
	end

endmodule