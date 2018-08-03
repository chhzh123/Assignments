`timescale 1ns / 1ps
module display (
	input [1:0] digit,
	input fb,
	output reg [6:0] seven_seg
	);

	// seven segments
	parameter zero	= 7'b000_0001;
	parameter one	= 7'b100_1111;
	parameter two	= 7'b001_0010;
	parameter three	= 7'b000_0110;
	parameter four	= 7'b100_1100;
	parameter five	= 7'b010_0100;
	parameter six	= 7'b010_0000;
	parameter seven	= 7'b000_1111;
	parameter eight	= 7'b000_0000;
	parameter nine	= 7'b000_0100;

	always @(digit)
	if (fb == 1)
	begin
		case(digit)
			0: seven_seg = four;
			1: seven_seg = three;
			2: seven_seg = seven;
			3: seven_seg = one;
		endcase // clk
	end
	else begin
	case(digit)
        0: seven_seg = five;
        1: seven_seg = one;
        2: seven_seg = zero;
        3: seven_seg = one;
    endcase // clk
	end

endmodule