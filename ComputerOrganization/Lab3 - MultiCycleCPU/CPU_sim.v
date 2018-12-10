`timescale 1ns / 1ps

module CPU_sim (
	output [2:0] state,
	output [1:0] RegDst,
	output ExtSel,
	output RegWrite, MemWrite,
	output ALUSrcA, ALUSrcB,
	output [2:0] ALUOp,
	output [1:0] PCSrc,
	output DBSrc,
	output WrRegSrc,
	output Zero,
	output PCWrite,
	output [31:0] currPC, nextPC, instruction, alu_res,
	output [4:0] rs, rt,
	output [31:0] d1, d2, rsData, rtData, dbData
	);
	
	reg clk;
	reg reset;

	CPU cpu(
		.clk(clk),
		.reset(reset),
		.state(state),
		.RegDst(RegDst),
		.ExtSel(ExtSel),
		.RegWrite(RegWrite),
		.MemWrite(MemWrite),
		.ALUSrcA(ALUSrcA),
		.ALUSrcB(ALUSrcB),
		.ALUOp(ALUOp),
		.DBSrc(DBSrc),
		.PCSrc(PCSrc),
		.WrRegSrc(WrRegSrc),
		.Zero(Zero),
		.PCWrite(PCWrite),
		.currPC(currPC),
		.nextPC(nextPC),
		.instruction(instruction),
		.alu_res(alu_res),
		.rs(rs),
		.rt(rt),
		.d1(d1),
		.d2(d2),
		.rsData(rsData),
		.rtData(rtData),
		.dbData(dbData)
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