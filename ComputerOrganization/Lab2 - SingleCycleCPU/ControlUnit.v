`timescale 1ns / 1ps

module ControlUnit (
    input [5:0] opcode,
    input Zero,
    output reg RegDst,
    output reg ExtSel,
    output reg RegWrite,
    output reg ALUSrcA,
    output reg ALUSrcB,
    output reg [2:0] ALUOp,
    output reg MemToReg,
    output reg MemWrite,
    output reg Branch,
    output reg Jump,
    output reg PCWrite
    );

    always @ (opcode or Zero) begin // Zero!
    	RegDst   <= 0;
		ExtSel   <= 0;
		RegWrite <= 1;
		ALUSrcA  <= 0;
		ALUSrcB  <= 0;
		ALUOp    <= 3'b000;
		MemToReg <= 0;
		MemWrite <= 0;
		Branch   <= 0;
		Jump     <= 0;
		PCWrite  <= 1;
    	case (opcode)
    		6'b000000: begin // add rd, rs, rt
    				RegDst <= 1;
    				end
    		6'b000001: begin // sub rd, rs, rt
    				RegDst <= 1;
    				ALUOp <= 3'b001;
    				end
    		6'b000010: begin // addiu rt, rs, imm
                    ExtSel <= 1; // ???
    				ALUSrcB <= 1;
    				end
    		6'b010000: begin // andi rt, rs, imm
    				ALUSrcB <= 1;
    				ALUOp <= 3'b100;
    				end
    		6'b010001: begin // and rd, rs, rt
    				RegDst <= 1;
    				ALUOp <= 3'b100;
    				end
    		6'b010010: begin // ori rt, rs, imm
    				ALUSrcB <= 1;
    				ALUOp <= 3'b011;
    				end
    		6'b010011: begin // or rd, rs, rt
    				RegDst <= 1;
    				ALUOp <= 3'b011;
    				end
    		6'b011000: begin // sll rd, rt, sa
    				RegDst <= 1;
    				ALUSrcA <= 1;
    				ALUOp <= 3'b010;
    				end
    		6'b011100: begin // slti rt, rs, imm
                    ExtSel <= 1;
                    ALUSrcB <= 1; // remember!
    				ALUOp <= 3'b110;
    				end
    		6'b100110: begin // sw rt, imm(rs)
    				ExtSel <= 1;
    				RegWrite <= 0;
    				ALUSrcB <= 1;
    				MemWrite <= 1;
    				end
    		6'b100111: begin // lw rt, imm(rs)
    				ExtSel <= 1;
    				ALUSrcB <= 1;
    				MemToReg <= 1;
    				end
    		6'b110000: begin // beq rs, rt, imm	
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b001;
    				Branch <= Zero;
					end
			6'b110001: begin // bne rs, rt, imm	
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b001;
    				Branch <= ~Zero; // (rs - rt == 0) ? 1 : 0 Not equal!
					end
			6'b110010: begin // bltz rs, imm
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b110; // compare sign
    				Branch <= ~Zero; // a < 0 ? 1 : 0
					end
			6'b111000: begin // j addr
    				RegWrite <= 0;
    				Jump <= 1;
					end
			6'b111111: begin // halt
					PCWrite <= 0;
					end
		endcase
	end
    
endmodule