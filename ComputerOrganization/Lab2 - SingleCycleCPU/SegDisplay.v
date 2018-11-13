`timescale 1ns / 1ps

module SegDisplay (
	input [3:0] data,
	output reg [6:0] dispcode
	);

	always @(data)
		case(data)
			// 0: on    1: off
			1: dispcode = 7'b100_1111;
			2: dispcode = 7'b001_0010;
			3: dispcode = 7'b000_0110;
			4: dispcode = 7'b100_1100;
			5: dispcode = 7'b010_0100;
			6: dispcode = 7'b010_0000;
			7: dispcode = 7'b000_1111;
			8: dispcode = 7'b000_0000;
			9: dispcode = 7'b000_0100;
			10: dispcode = 7'b000_1000; // A
			11: dispcode = 7'b110_0000; // b
			12: dispcode = 7'b011_0001; // C
			13: dispcode = 7'b100_0010; // d
			14: dispcode = 7'b001_0000; // e
			15: dispcode = 7'b011_1000; // F
		endcase

endmodule