`timescale 1ns / 1ps

module Show(
	input clk,
	input clk_cpu, // button
	input reset,
	input [1:0] SW_in,
	input State_in,
	output reg [6:0] dispcode,
	output reg [3:0] out
	);

	// synchronize and reduce jitter
	reg in_detected = 1'b0;
	reg [15:0] inhistory = 16'h0000;
	always @(posedge clk) begin
		inhistory = {inhistory[15:0], clk_cpu};
		if (inhistory == 16'b0011111111111111)
			in_detected <= 1'b1;
		else
			in_detected <= 1'b0;
	end

	wire [1:0] seg_num; // not reg!

	Counter counter(
		.clk(clk),
		// output clock/counter
		.count_4(seg_num)
		);

	reg [31:0] firstNum;
	reg [31:0] secondNum;

	initial firstNum = 0;
	initial secondNum = 0;

	wire [31:0] currPC, nextPC, rsData, rtData, dbData, alu_res;
	wire [4:0] rs, rt;
	wire [2:0] state;

	CPU cpu(
		// input
		.clk(in_detected),
		.reset(reset),
		// output
		.currPC(currPC),
		.nextPC(nextPC),
		.rs(rs),
		.rt(rt),
		.rsData(rsData),
		.rtData(rtData),
		.dbData(dbData),
		.alu_res(alu_res),
		.state(state)
		);

	always @(SW_in or State_in) begin
		if (State_in == 0)
		case (SW_in)
			2'b00: begin
				firstNum <= currPC;
				secondNum <= nextPC;
				end
			2'b01: begin
				firstNum <= {{27{1'b0}},rs};
				secondNum <= rsData;
				end
			2'b10: begin
				firstNum <= {{27{1'b0}},rt};
				secondNum <= rtData;
				end
			2'b11: begin
				firstNum <= alu_res;
				secondNum <= dbData;
				end
		endcase
		else begin
			firstNum <= 0;
			secondNum <= {{29{1'b0}},state[2:0]};
		end
	end

	SegDisplay seg1(
		.data(firstNum[7:4])
		// .dispcode
		);

	SegDisplay seg2(
		.data(firstNum[3:0])
		// .dispcode
		);

	SegDisplay seg3(
		.data(secondNum[7:4])
		// .dispcode
		);

	SegDisplay seg4(
		.data(secondNum[3:0])
		// .dispcode
		);

	always @ (seg_num or firstNum or secondNum)
		case (seg_num)
			0: begin
				out = 4'b1110;
				dispcode = seg4.dispcode;
				end
			1: begin
				out = 4'b1101;
				dispcode = seg3.dispcode;
				end
			2: begin
				out = 4'b1011;
				dispcode = seg2.dispcode;
				end
			3: begin
				out = 4'b0111;
				dispcode = seg1.dispcode;
				end
		endcase

endmodule