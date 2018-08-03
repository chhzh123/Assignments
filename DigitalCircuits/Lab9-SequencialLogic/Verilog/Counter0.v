`timescale 1ns / 1ps
module Counter(
    input clr, // clear, say reset
    input wire clk, // clock
    input wire fb, // front or back
    output [6:0] seg,
    output reg [1:0] an
    );
    
    // display 10kHz
    reg [16:0] count_dis; // 26 bits to store count: 2^17 > 10^5
    reg clk_dis;
    always @ (posedge clk or posedge clr)
    begin
        if (clr == 1) // reset
        begin
            clk_dis <= 0;
            count_dis <= 0;
        end
        else if (count_dis == 50_000 - 1) // return 0
        begin
            clk_dis <= ~clk_dis;
            count_dis <= 0;
        end
        else
        begin
            clk_dis <= clk_dis;
            count_dis <= count_dis + 1;
        end
    end

    reg [1:0] out;
    always @ (posedge clk_dis or posedge clr)
    begin
        if (clr == 1 || out == 4)
            out <= 0;
        else
            out <= out + 1;
    end

    // time counter + frequency divisor (flip-flop)
    // 1s
    localparam MAX_COUNT_FEQ = 50_000_000; // 0.5s
    reg [25:0] count_ns; // 26 bits to store count: 2^26 > 10^5
    reg clk_sec;
    always @ (posedge clk or posedge clr)
    begin
        if (clr == 1) // reset
        begin
            clk_sec <= 0;
            count_ns <= 0;
        end
        else if (count_ns == MAX_COUNT_FEQ - 1) // return 0
        begin
            clk_sec <= ~clk_sec;
            count_ns <= 0;
        end
        else
        begin
            clk_sec <= clk_sec;
            count_ns <= count_ns + 1;
        end
    end

    // 10s time counter
    reg [3:0] count_sec_1;
    always @ (posedge clk_sec or posedge clr)
    begin
        if (clr == 1)
            count_sec_1 <= 0;
        else if (count_sec_1 == 10 - 1) // next stage
            count_sec_1 <= 0;
        else
            count_sec_1 <= count_sec_1 + 1;
    end

    // 1min
    reg [3:0] count_sec_2;
    always @ (count_sec_1)
    begin
        if (clr == 1)
            count_sec_2 <= 0;
        else if (count_sec_1 + 1 == 10)
                if (count_sec_2 + 1 == 6)
                    count_sec_2 <= 0;
                else
                    count_sec_2 <= count_sec_2 + 1;
            else
                count_sec_2 <= count_sec_2;
    end

    // 10min
    reg [3:0] count_min_1;
    always @ (count_sec_2)
    begin
        if (clr == 1)
            count_min_1 <= 0;
        else if (count_sec_2 + 1 == 6)
                if (count_min_1 + 1 == 10)
                    count_min_1 <= 0;
                else
                    count_min_1 <= count_min_1 + 1;
            else
                count_min_1 <= count_min_1;
    end

    // 99min
    reg [3:0] count_min_2;
    always @ (count_min_1)
    begin
        if (clr == 1)
            count_min_2 <= 0;
        else if (count_min_1 + 1 == 10)
                if (count_min_2 + 1 == 10)
                    count_min_2 <= 0;
                else
                    count_min_2 <= count_min_2 + 1;
            else
                count_min_2 <= count_min_2;
    end

    display disp1 (.dis(out),
                   .count_sec_1(count_sec_1),
                   .count_sec_2(count_sec_2),
                   .count_min_1(count_min_1),
                   .count_min_2(count_min_2),
                   .seven_seg(seg));

    always @ (out)
        case (out)
            0: an = 4'b1110;
            1: an = 4'b1101;
            2: an = 4'b1011;
            3: an = 4'b0111;
        endcase

endmodule