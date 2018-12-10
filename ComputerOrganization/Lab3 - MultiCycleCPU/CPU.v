`timescale 1ns / 1ps

module CPU (
	input clk, reset,
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
	output wire [31:0] d1, d2, rsData, rtData, dbData,
	output wire [4:0] rs, rt, rd, sa
	);

	wire [5:0] opcode;
	wire [15:0] imm;
	wire [31:0] pc4;
	wire [31:0] inst;

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
		.dataOut(inst)
		);

	Register ir(
		.clk(clk),
		.reset(reset),
		.in(inst),
		.out(instruction)
		);

	// instruction decode (ID)
	ControlUnit control(
		.clk(clk),
		.reset(reset),
		// input
		.opcode(opcode),
		.Zero(Zero),
		// output
		.state(state),
		.RegDst(RegDst),
		.ExtSel(ExtSel),
		.RegWrite(RegWrite),
		.ALUSrcA(ALUSrcA),
		.ALUSrcB(ALUSrcB),
		.ALUOp(ALUOp),
		.DBSrc(DBSrc),
		.WrRegSrc(WrRegSrc),
		.MemWrite(MemWrite),
		.PCSrc(PCSrc),
		.PCWrite(PCWrite)
		);

	RegFile reg_file(
		.clk(clk),
		.reset(reset),
		.r1(rs),
		.r2(rt),
		.wr(mux_regdst.res),
		.RegWrite(RegWrite),
		.wd(mux_wrreg.res)
		// d1 -> adr / mux_pc.D
		// d2 -> bdr
		);

	Register adr(
		.clk(clk),
		.reset(reset),
		.in(reg_file.d1)
		// out -> mux_srcA.A
		);

	Register bdr(
		.clk(clk),
		.reset(reset),
		.in(reg_file.d2)
		// out -> mux_srcB.A / dm.dataIn
		);

	assign d1 = mux_srcA.res;
	assign d2 = mux_srcB.res;
	assign rsData = reg_file.d1;
	assign rtData = reg_file.d2;

	// execution (EXE)
	MUX5b mux_regdst(
		.Sel(RegDst),
		.A(rt),
		.B(rd),
		.C(5'b11111)
		// res -> reg_file.wr
		);

	MUX mux_srcA(
		.Sel(ALUSrcA),
		.A(adr.out),
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
		.A(bdr.out),
		.B(extender.dataOut)
		// res -> alu.B
		);

	ALU alu(
		.op(ALUOp),
		.A(mux_srcA.res),
		.B(mux_srcB.res),
		// res -> aludr / mux_memToReg
		.zero(Zero)
		);

	Register aludr(
		.clk(clk),
		.reset(reset),
		.in(alu.res),
		// res -> dm.address / mux_memToReg.A
		.out(alu_res)
		);

	// access memory (MEM)
	DataMemory dm(
		.clk(clk),
		.reset(reset),
		.address(alu_res),
		.MemWrite(MemWrite),
		.dataIn(bdr.out)
		// dataOut -> mux_memToReg
		);

	// write back (WB)
	MUX mux_memToReg(
		.Sel(DBSrc),
		.A(alu.res), // not alu_res!
		.B(dm.dataOut)
		// res -> dbdr
		);

	Register dbdr(
		.clk(clk),
		.reset(reset),
		.in(mux_memToReg.res)
		// out -> mux_wrreg.B
		);

	MUX mux_wrreg(
		.Sel(WrRegSrc),
		.A(dbdr.out),
		.B(pc4)
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

	MUX4 mux_pc(
		.Sel(PCSrc),
		.A(pc4),
		.B({pc4[31:28],instruction[25:0],2'b00}),
		.C(add_target.res),
		.D(reg_file.d1),
		.res(nextPC)
		);

endmodule