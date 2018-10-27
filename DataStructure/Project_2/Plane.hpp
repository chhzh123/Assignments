//
//  Plane.hpp
//  AirportSim
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#ifndef PLANE_HPP
#define PLANE_HPP

#include <iostream>
using namespace std;

enum Status {null, arriving, departing};

class Plane{
public:
    Plane(){
        flt_num = -1;
        clock_start = -1;
        state = Status::null;
    }
    Plane(int flt,int time,Status status,int remained_time=-1){
        flt_num = flt;
        clock_start = time;
        state = status;
        if (remained_time == -1){
            fuel = 0x3f3f3f3f;
            cout << "Plane number " << flt << " ready to ";
            if(status == Status::arriving)
                cout << "land." << endl;
            else
                cout << "take off." << endl;
        } else {
            fuel = rand() % (remained_time+1);
            fuel = (fuel == 0 ? 1 : fuel);
            cout << "Plane number " << flt << " ready to ";
            if(status == Status::arriving)
                cout << "land. Fuel: " << fuel << endl;
            else
                cout << "take off. Fuel: " << fuel << endl;
        }
    }
    int get_fuel() const{
        return fuel;
    }
    void reduce_fuel(){
        fuel--; // can set other functions
    }
    void refuse() const;
    void land(int time,int rw_num) const;
    void fly(int time,int rw_num) const;
    void crash() const;
    int started() const;
    
private:
    int flt_num;
    int fuel;
    int clock_start;
    Status state;
};

void Plane::crash() const{
    cout << "Plane number " << flt_num << " crashed!" << endl;
}

void Plane::refuse() const{
    cout << "Plane number " << flt_num;
    if (state == Status::arriving)
        cout << " directed to another airport!" << endl;
    else
        cout << " told to try to takeoff again later." << endl;
    // broadcast refuse
}

void Plane::land(int time,int rw_num=0) const{
    // state == Status::arriving
    int wait = time - clock_start;
    cout << time << ": Plane number " << flt_num << " landed on runway #" << rw_num << " after " << wait
         << " time unit" << ((wait <= 1) ? "" : "s")
         << " in the landing queue." << endl;
    // broadcast land & land time
}

void Plane::fly(int time,int rw_num=0) const{
    // state == Status::departing
    int wait = time - clock_start;
    cout << time << ": Plane number " << flt_num << " took off from runway #" << rw_num << " after " << wait
         << " time unit" << ((wait <= 1) ? "" : "s")
         << " in the takeoff queue." << endl;
    // broadcast takeoff & takeoff time
}

int Plane::started() const{
    return clock_start;
}

#endif // PLANE_HPP