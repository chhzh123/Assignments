`timescale 1ns / 1ps
module display (
	input [1:0] dis,
	input [3:0] count_sec_1, // 2^4 > 10
	input [3:0] count_sec_2, // 2^3 > 6 still use 4 bits
	input [3:0] count_min_1,
	input [3:0] count_min_2,
	output reg [6:0] seven_seg
	);

	// seven segments
	parameter _0	= 7'b000_0001;
	parameter _1	= 7'b100_1111;
	parameter _2	= 7'b001_0010;
	parameter _3	= 7'b000_0110;
	parameter _4	= 7'b100_1100;
	parameter _5	= 7'b010_0100;
	parameter _6	= 7'b010_0000;
	parameter _7	= 7'b000_1111;
	parameter _8	= 7'b000_0000;
	parameter _9	= 7'b000_0100;

	always @ (dis)
	case (dis)
		0: // units (second)
		case (count_sec_1)
			0: seven_seg = _0;
			1: seven_seg = _1;
			2: seven_seg = _2;
			3: seven_seg = _3;
			4: seven_seg = _4;
			5: seven_seg = _5;
			6: seven_seg = _6;
			7: seven_seg = _7;
			8: seven_seg = _8;
			9: seven_seg = _9;
		endcase // count
		1: // tens (second)
		case (count_sec_2)
			0: seven_seg = _0;
			1: seven_seg = _1;
			2: seven_seg = _2;
			3: seven_seg = _3;
			4: seven_seg = _4;
			5: seven_seg = _5;
			6: seven_seg = _6;
			7: seven_seg = _7;
			8: seven_seg = _8;
			9: seven_seg = _9;
		endcase // count
		2: // units (minute)
		case (count_min_1)
			0: seven_seg = _0;
			1: seven_seg = _1;
			2: seven_seg = _2;
			3: seven_seg = _3;
			4: seven_seg = _4;
			5: seven_seg = _5;
			6: seven_seg = _6;
			7: seven_seg = _7;
			8: seven_seg = _8;
			9: seven_seg = _9;
		endcase // count
		3: // tens (minute)
		case (count_min_2)
			0: seven_seg = _0;
			1: seven_seg = _1;
			2: seven_seg = _2;
			3: seven_seg = _3;
			4: seven_seg = _4;
			5: seven_seg = _5;
			6: seven_seg = _6;
			7: seven_seg = _7;
			8: seven_seg = _8;
			9: seven_seg = _9;
		endcase // count
	endcase // dis

endmodule