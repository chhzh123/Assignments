`timescale 1ns / 1ps

module CPU (
	input clk, reset,
	output RegDst,
	output ExtSel,
	output RegWrite, MemWrite,
	output ALUSrcA, ALUSrcB,
	output [2:0] ALUOp,
	output MemToReg,
	output Branch, Jump, Zero,
	output PCWrite,
	output [31:0] currPC, nextPC, instruction, alu_res,
	output wire [31:0] d1, d2, rsData, rtData, dbData,
	output wire [4:0] rs, rt, rd, sa
	);

	wire [5:0] opcode;
	wire [15:0] imm;
	wire [31:0] pc4;

	assign opcode = instruction[31:26];
	assign rs = instruction[25:21];
	assign rt = instruction[20:16];
	assign rd = instruction[15:11];
	assign sa = instruction[10:6];
	assign imm = instruction[15:0];

	PC pc(
		.clk(clk),
		.reset(reset),
		.PCWrite(PCWrite),
		.currPC(currPC),
		.nextPC(nextPC)
		);

	Adder add_pc4(
		.A(currPC),
		.B({{29{1'b0}},3'b100}),
		.res(pc4)
		);
	
	// instruction fetch (IF)
	InstructionMemory im(
		.address(currPC),
		.dataOut(instruction)
		);

	// instruction decode (ID)
	ControlUnit control(
		// input
		.opcode(opcode),
		.Zero(Zero),
		// output
		.RegDst(RegDst),
		.ExtSel(ExtSel),
		.RegWrite(RegWrite),
		.ALUSrcA(ALUSrcA),
		.ALUSrcB(ALUSrcB),
		.ALUOp(ALUOp),
		.MemToReg(MemToReg),
		.MemWrite(MemWrite),
		.Branch(Branch),
		.Jump(Jump),
		.PCWrite(PCWrite)
		);

	// execution (EXE)
	Registers reg_file(
		.clk(clk),
		.reset(reset),
		.r1(rs),
		.r2(rt),
		.wr(mux_regdst.res),
		.RegWrite(RegWrite),
		.wd(mux_memToReg.res)
		// d1 -> mux_srcA.A
		// d2 -> mux_srcB.A / dm.dataIn
		);

	assign d1 = mux_srcA.res;
	assign d2 = mux_srcB.res;
	assign rsData = reg_file.d1;
	assign rtData = reg_file.d2;

	MUX5 mux_regdst(
		.Sel(RegDst),
		.A(rt),
		.B(rd)
		// res -> reg_file.wr
		);

	MUX mux_srcA(
		.Sel(ALUSrcA),
		.A(reg_file.d1),
		.B({{27{1'b0}},sa})
		// res -> alu.A
		);
	
	Extender extender(
		.Sel(ExtSel),
		.dataIn(imm)
		// dataOut -> mux_srcB.B / sl_16
		);

	MUX mux_srcB(
		.Sel(ALUSrcB),
		.A(reg_file.d2),
		.B(extender.dataOut)
		// res -> alu.B
		);

	ALU alu(
		.op(ALUOp),
		.A(mux_srcA.res),
		.B(mux_srcB.res),
		// res -> dm.address / mux_memToReg.A
		.res(alu_res),
		.zero(Zero)
		);

	// access memory (MEM)
	DataMemory dm(
		.clk(clk),
		.reset(reset),
		.address(alu_res),
		.MemWrite(MemWrite),
		.dataIn(reg_file.d2)
		// dataOut -> mux_memToReg
		);

	// write back (WB)
	MUX mux_memToReg(
		.Sel(MemToReg),
		.A(alu_res),
		.B(dm.dataOut)
		// res -> reg_file.wd
		);

	assign dbData = mux_memToReg.res;

	// jump & branch
	ShiftLeft sl(
		.dataIn(extender.dataOut)
		// dataOut -> add_target.B
		);

	Adder add_target(
		.A(pc4),
		.B(sl.dataOut)
		// res -> mux_branch.B
		);

	MUX mux_branch(
		.Sel(Branch),
		.A(pc4),
		.B(add_target.res)
		// res -> mux_jump.A
		);

	MUX mux_jump(
		.Sel(Jump),
		.A(mux_branch.res),
		.B({pc4[31:28],instruction[25:0],2'b00}),
		.res(nextPC)
		);

endmodule