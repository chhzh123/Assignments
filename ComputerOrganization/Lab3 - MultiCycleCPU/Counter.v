`timescale 1ns / 1ps

module Counter(
    input clr, // clear, say reset
    input clk, // original clock
    output reg [1:0] count_4,
    output reg clk_seg
    );
    
    // display 10kHz
    reg [16:0] count_dis; // 26 bits to store count: 2^17 > 10^5
    always @ (posedge clk or posedge clr)
    begin
        if (clr == 1) // reset
        begin
            clk_seg <= 0;
            count_dis <= 0;
        end
        else if (count_dis == 50_000 - 1) // return 0
        begin
            clk_seg <= ~clk_seg;
            count_dis <= 0;
        end
        else
        begin
            clk_seg <= clk_seg;
            count_dis <= count_dis + 1;
        end
    end

    always @ (posedge clk_seg or posedge clr)
    begin
        if (clr == 1 || count_4 == 4)
            count_4 <= 0;
        else
            count_4 <= count_4 + 1;
    end

endmodule