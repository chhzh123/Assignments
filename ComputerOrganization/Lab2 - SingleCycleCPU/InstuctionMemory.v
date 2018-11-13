`timescale 1ns / 1ps

module InstructionMemory (
	input [31:0] address,
	output [31:0] dataOut
	);
	
	reg [7:0] memory [0:255];

	// initialization
	initial $readmemb("D:/instruction.data",memory);
	// $display("Read in data successfully!");

	// output data (little endian)
	assign dataOut = {memory[address + 3],
					  memory[address + 2],
					  memory[address + 1],
					  memory[address]};

endmodule