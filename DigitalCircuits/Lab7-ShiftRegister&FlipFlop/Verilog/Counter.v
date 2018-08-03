`timescale 1ns / 1ps
// 100MHz -> 1kHz
module Counter(
    input clr, // clear, say reset
    input wire clk, // clock
    input wire fb, // front or back
    output [6:0] seg,
    output reg [3:0] an
    );

    reg [1:0] out;
    parameter MAX_COUNT = 4;
    
    // time counter
    localparam MAX_COUNT_TIME = 100_000; // 10kHz
    reg [16:0] count; // 26 bits to store count: 2^17 > 10^5
    always @ (posedge clk or posedge clr)
    begin
        if (clr == 1) // reset
            count <= 0;
        else if (count == MAX_COUNT_TIME - 1) // return 0
            count <= 0;
        else
            count <= count + 1;
    end

    // frequency divisor (flip-flop)
    reg clk_div;
    always @ (posedge clk, posedge clr)
    begin
        if (clr == 1)
            clk_div <= 0;
        else if (count == MAX_COUNT - 1) // reset
            clk_div <= ~clk_div;
        else // set
            clk_div <= clk_div;
    end

    always @ (posedge clk_div or posedge clr)
    begin
        if (clr == 1)
            out <= 0;
        if (out == MAX_COUNT)
            out <= 0;
        else
            out <= out +1;
    end

    display disp1 (.digit(out),.seven_seg(seg),.fb(fb));
    
    always @ (out)
        case(out)
            0: an = 4'b1110;
            1: an = 4'b1101;
            2: an = 4'b1011;
            3: an = 4'b0111;
        endcase



    // hex counter
    // always @ (posedge clk_div, posedge clr)
    // begin
    //     if (clr == 1)
    //         out <= 0;
    //     else if (clk_div == 1)
    //         out <= out + 1;
    //     else
    //         out <= out;
    // end

endmodule
