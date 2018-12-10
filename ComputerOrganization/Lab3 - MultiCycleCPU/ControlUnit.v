`timescale 1ns / 1ps

module ControlUnit (
    input clk,
    input reset,
    input [5:0] opcode,
    input Zero,
    output reg [2:0] state,
    output reg [1:0] RegDst,
    output reg ExtSel,
    output reg ALUSrcA,
    output reg ALUSrcB,
    output reg [2:0] ALUOp,
    output reg [1:0] PCSrc,
    output reg DBSrc,
    output reg WrRegSrc,
    output reg RegWrite,
    output reg MemWrite,
    output reg PCWrite
    );

    initial state = 3'b000;

    // Finite State Machine (FSM)
    always @ (posedge clk) begin
        if (reset == 0)
            state <= 3'b000;
        else case (state)
            3'b000: begin // IF
                state <= 3'b001;
            end
            3'b001: begin // ID
                if (opcode == 6'b110100 || opcode == 6'b110101 || opcode == 6'b110110) // beq bne bltz
                    state <= 3'b101;
                else if (opcode == 6'b110000 || opcode == 6'b110001) // sw & lw
                    state <= 3'b010;
                else if (opcode == 6'b111000 || opcode == 6'b111001 || opcode == 6'b111010 || opcode == 6'b111111) // j jr jal halt
                    state <= 3'b000;
                else
                    state <= 3'b110;
            end
            3'b110: begin // EXE
                state <= 3'b111;
            end
            3'b111: begin // WB
                state <= 3'b000;
            end
            3'b101: begin // EXE
                state <= 3'b000;
            end
            3'b010: begin // EXE
                state <= 3'b011;
            end
            3'b011: begin // MEM
                if (opcode == 6'b110001) // lw
                    state <= 3'b100;
                else // sw
                    state <= 3'b000;
            end
            3'b100: begin // WB
                state <= 3'b000;
            end
        endcase
    end

    // always @ (opcode or state or Zero) begin // Zero!
    always @ (state or opcode) begin
    	RegDst   <= 2'b00;
		ExtSel   <= 0;
        if (state == 3'b111 || state == 3'b100)
		    RegWrite <= 1;
        else
            RegWrite <= 0;
		ALUSrcA  <= 0;
		ALUSrcB  <= 0;
		ALUOp    <= 3'b000;
		DBSrc <= 0;
		MemWrite <= 0;
        PCSrc    <= 2'b00;
		WrRegSrc <= 0;
        if (state == 3'b000)
		    PCWrite <= 1;
        else
            PCWrite <= 0; // donot update PC if the state is not IF!
    	case (opcode)
    		6'b000000: begin // add rd, rs, rt
    				RegDst <= 2'b01;
    				end
    		6'b000001: begin // sub rd, rs, rt
    				RegDst <= 2'b01;
    				ALUOp <= 3'b001;
    				end
    		6'b000010: begin // addiu rt, rs, imm
                    ExtSel <= 1; // ???
    				ALUSrcB <= 1;
    				end
            6'b010000: begin // and rd, rs, rt
                    RegDst <= 2'b01;
                    ALUOp <= 3'b100;
                    end
    		6'b010001: begin // andi rt, rs, imm
    				ALUSrcB <= 1;
    				ALUOp <= 3'b100;
    				end
    		6'b010010: begin // ori rt, rs, imm
    				ALUSrcB <= 1;
    				ALUOp <= 3'b011;
    				end
    		6'b010011: begin // xori rt, rs, imm
                    ALUSrcB <= 1;
                    ALUOp <= 3'b111;
    				end
    		6'b011000: begin // sll rd, rt, sa
    				RegDst <= 2'b01;
    				ALUSrcA <= 1;
    				ALUOp <= 3'b010;
    				end
    		6'b100110: begin // slti rt, rs, imm
                    ExtSel <= 1;
                    ALUSrcB <= 1; // remember!
    				ALUOp <= 3'b110;
    				end
            6'b100111: begin // slt rd, rs, rt
                    RegDst <= 2'b01;
                    ALUOp <= 3'b110;
                    end
    		6'b110000: begin // sw rt, imm(rs)
    				ExtSel <= 1;
    				RegWrite <= 0;
    				ALUSrcB <= 1;
                    if (state == 3'b011)
    				    MemWrite <= 1;
                    else
                        MemWrite <= 0;
    				end
    		6'b110001: begin // lw rt, imm(rs)
    				ExtSel <= 1;
    				ALUSrcB <= 1;
    				DBSrc <= 1;
    				end
    		6'b110100: begin // beq rs, rt, imm	
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b001;
                    if (Zero == 1)
                        PCSrc <= 2'b10;
                    else
                        PCSrc <= 2'b00;
					end
			6'b110101: begin // bne rs, rt, imm	
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b001;
                    if (Zero == 0) // (rs - rt == 0) ? 1 : 0 Not equal!
                        PCSrc <= 2'b10;
                    else
                        PCSrc <= 2'b00;
					end
			6'b110110: begin // bltz rs, imm
					ExtSel <= 1;
    				RegWrite <= 0;
    				ALUOp <= 3'b110; // compare sign
                    if (Zero == 0) // a < 0 ? 1 : 0
                        PCSrc <= 2'b10;
                    else
                        PCSrc <= 2'b00;
					end
			6'b111000: begin // j addr
    				RegWrite <= 0;
    				PCSrc <= 2'b01;
					end
            6'b111001: begin // jr rs
                    RegWrite <= 0;
                    PCSrc <= 2'b11;
                    end
            6'b111010: begin // jal addr
                    if (state == 3'b001)
                        RegWrite <= 1;
                    RegDst <= 2'b10;
                    WrRegSrc <= 1;
                    PCSrc <= 2'b01;
                    end
			6'b111111: begin // halt
					PCWrite <= 0;
					end
		endcase
	end
    
endmodule