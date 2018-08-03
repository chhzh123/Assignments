`timescale 1ns / 1ps
module display (
	input [3:0] digit,
	output reg [6:0] seven_seg
	);

	always @(digit)
		case(digit)
			1: seven_seg = 7'b100_1111;
			2: seven_seg = 7'b001_0010;
			3: seven_seg = 7'b000_0110;
			4: seven_seg = 7'b100_1100;
			5: seven_seg = 7'b010_0100;
			6: seven_seg = 7'b010_0000;
			7: seven_seg = 7'b000_1111;
			8: seven_seg = 7'b000_0000;
			9: seven_seg = 7'b000_0100;
			10: seven_seg = 7'b000_1000; // a
			11: seven_seg = 7'b110_0000; // b
			12: seven_seg = 7'b011_0001; // c
		endcase // clk

endmodule