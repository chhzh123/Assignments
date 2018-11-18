`timescale 1ns / 1ps

module CPU_sim (
	output RegDst,
	output ExtSel,
	output RegWrite, MemWrite,
	output ALUSrcA, ALUSrcB,
	output [2:0] ALUOp,
	output MemToReg,
	output Branch, Jump, Zero,
	output PCWrite,
	output [31:0] currPC, nextPC, instruction, alu_res,
	output wire [31:0] d1, d2
	);
	
	reg clk;
	reg reset;

	CPU cpu(
		.clk(clk),
		.reset(reset),
		.RegDst(RegDst),
		.ExtSel(ExtSel),
		.RegWrite(RegWrite),
		.MemWrite(MemWrite),
		.ALUSrcA(ALUSrcA),
		.ALUSrcB(ALUSrcB),
		.ALUOp(ALUOp),
		.MemToReg(MemToReg),
		.Branch(Branch),
		.Jump(Jump),
		.Zero(Zero),
		.PCWrite(PCWrite),
		.currPC(currPC),
		.nextPC(nextPC),
		.instruction(instruction),
		.alu_res(alu_res),
		.d1(d1),
		.d2(d2)
		);

	initial begin
		clk = 1;
		reset = 0;
		// wait for initialization
		#30;
		reset = 1;
	end

	always #50 clk = ~clk;

endmodule