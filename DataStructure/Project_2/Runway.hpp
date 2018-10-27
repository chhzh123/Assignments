//
//  Runway.hpp
//  AirportSim
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#ifndef RUNWAY_HPP
#define RUNWAY_HPP

#include <vector>
#include <algorithm>
#include "Plane.hpp"
using namespace std;

//bool value => Error_Code true: success false: fail
enum Runway_activity {idle, land, takeoff};
enum Runway_mode {mainly_landing, mainly_takeoff, only_landing, only_takeoff};

class Runway {
    queue<Plane> landing;
    queue<Plane> takeoff;
    vector<Plane> landing_f;
    int num_runways;
    vector<Runway_mode> runway_mode;
    int num_landing_runways;
    int num_takeoff_runways;

    int queue_limit;
    int num_land_request;
    int num_takeoff_request;
    int num_landings;
    int num_takeoffs;
    int num_land_accepted;
    int num_takeoff_accepted;
    int num_land_refused;
    int num_takeoff_refused;
    int num_crashed;
    int land_wait;
    int takeoff_wait;
    int idle_time;

    Runway_activity land_one_plane(int time,int rw_num);
    Runway_activity takeoff_one_plane(int time,int rw_num);
    
public:
    Runway(int limit){
        queue_limit = limit;
        num_runways = num_landing_runways = num_takeoff_runways = 1;
        runway_mode.push_back(Runway_mode::mainly_landing);
        num_land_request = num_takeoff_request = num_landings
            = num_takeoffs = num_land_accepted = num_takeoff_accepted 
            = num_land_refused = num_takeoff_refused = land_wait 
            = takeoff_wait = idle_time = num_crashed = 0;
    }
    void set_runways(int num_rw, vector<Runway_mode> rw_mode){
        num_runways = num_rw;
        runway_mode = rw_mode;
        num_landing_runways = num_takeoff_runways = 0;
        for (auto runway: runway_mode){
            if (runway != Runway_mode::only_takeoff)
                num_landing_runways++;
            if (runway != Runway_mode::only_landing)
                num_takeoff_runways++;
        }
    }

    bool landing_full() const{
        return (landing.size() >= queue_limit);
    }
    bool takeoff_empty() const{
        return takeoff.empty();
    }
    bool landing_empty() const{
        return landing.empty();
    }
    bool Qlanding_runway() const{
        return (num_landing_runways != 0);
    }
    bool Qtakeoff_runway() const{
        return (num_takeoff_runways != 0);
    }

    bool can_land(const Plane &current,int mode=0){ // mode 0: without considering fuel
        bool result;
        if (landing.size() < queue_limit && Qlanding_runway()) {
            result = true;
            if (mode == 0)
                landing.push(current);
            else
                landing_f.push_back(current);
        } else {
            result = false;
        }
        num_land_request++;
        if (result != true) {
            num_land_refused++;
        } else {
            num_land_accepted++;
        }
        return result;
    }
    
    bool can_depart(const Plane &current){
        bool result;
        if (takeoff.size() < queue_limit && Qtakeoff_runway()) {
            result = true;
            takeoff.push(current);
        } else {
            result = false;
        }
        num_takeoff_request++;
        if (result != true) {
            num_takeoff_refused++;
        } else {
            num_takeoff_accepted++;
        }
        return result;
    }
    
    void run_idle(int time, int runway_num = 1)
    {
        idle_time++;
        cout << time << ": Runway #" << runway_num << " is idle." << endl;
    }

    void change_runway_status(int runway_num, Runway_mode rw_mode);

    // modify the original activity
    void activity(int time,int mode);
    
    void shut_down(int time) const;
};

void Runway::change_runway_status(int runway_num, Runway_mode rw_mode)
{
    if (runway_mode[runway_num] != Runway_mode::only_takeoff)
        num_landing_runways--;
    if (runway_mode[runway_num] != Runway_mode::only_landing)
        num_takeoff_runways--;
    runway_mode[runway_num] = rw_mode;
    if (runway_mode[runway_num] != Runway_mode::only_takeoff)
        num_landing_runways++;
    if (runway_mode[runway_num] != Runway_mode::only_landing)
        num_takeoff_runways++;
}

Runway_activity Runway::land_one_plane(int time,int rw_num=0)
{
    Plane moving = landing.front();
    land_wait += time - moving.started();
    num_landings++;
    moving.land(time,rw_num); // for plane
    landing.pop();
    return Runway_activity::land;
}

Runway_activity Runway::takeoff_one_plane(int time,int rw_num=0)
{
    Plane moving = takeoff.front();
    takeoff_wait += time - moving.started();
    num_takeoffs++;
    moving.fly(time,rw_num); // for plane
    takeoff.pop();
    return Runway_activity::takeoff;
}

void Runway::activity(int time,int mode=0) // mode 0: without considering fuel
{
    if (mode != 0)
    {
        if (!landing_f.empty())
        {
            sort(landing_f.begin(),landing_f.end(),[](const Plane& plane1, const Plane& plane2)
                {return plane1.get_fuel() < plane2.get_fuel();});
            int index = 0;
            for (auto plane = landing_f.begin(); plane != landing_f.end(); ++plane)
                if (plane->get_fuel() < 0) {
                    index++;
                    plane->crash();
                    num_crashed++;
                } else if (plane->get_fuel() == 1) {
                    cout << "*** A PLANE DECLARED EMERGENCY *** " << endl;
                    land_wait += time - plane->started();
                    plane->land(time);
                    num_landings++;
                    index++; // no break!
                } else {
                    land_wait += time - plane->started();
                    plane->land(time);
                    num_landings++;
                    index++;
                    break;
                }
            // remove the crashed and landed plane
            for (int i = 0; i < index; ++i)
                landing_f.erase(landing_f.begin());
            // reduce the remaining planes' fuel
            for (auto plane = landing_f.begin(); plane != landing_f.end(); ++plane)
                plane->reduce_fuel();
        }
        else if (!takeoff.empty())
            takeoff_one_plane(time);
        else
            run_idle(time,0);
        return;
    }
    Runway_activity in_progress = Runway_activity::idle;
    for (int i = 0; i < num_runways; ++i)
    {
        switch (runway_mode[i])
        {
            case mainly_landing:
                if (!landing.empty())
                    in_progress = land_one_plane(time,i);
                else if (!takeoff.empty())
                    in_progress = takeoff_one_plane(time,i);
                break;

            case mainly_takeoff:
                if (!takeoff.empty())
                    in_progress = takeoff_one_plane(time,i);
                else if (!landing.empty())
                    in_progress = land_one_plane(time,i);
                break;

            case only_landing:
                if (!landing.empty())
                    in_progress = land_one_plane(time,i);
                break;

            case only_takeoff:
                if (!takeoff.empty())
                    in_progress = takeoff_one_plane(time,i);
                break;
        }
        // current runway does nothing
        if (in_progress == Runway_activity::idle)
            run_idle(time,i);
    }
}

void Runway::shut_down(int time) const
{
    cout << "Simulation has concluded after " << time << " time units." << endl
    << "Total number of planes processed: "
    << (num_land_request + num_takeoff_request) << endl
    << "Total number of planes asking to land: "
    << num_land_request << endl
    << "Total number of planes asking to take off: "
    << num_takeoff_request << endl
    << "Total number of planes accepted for landing: "
    << num_land_accepted << endl
    << "Total number of planes accepted for take off: "
    << num_takeoff_accepted << endl
    << "Total number of planes refused for landing: "
    << num_land_refused << endl
    << "Total number of planes refused for take off: "
    << num_takeoff_refused << endl
    << "Total number of planes that crashed: "
    << num_crashed << endl
    << "Total number of planes that landed: "
    << num_landings << endl
    << "Total number of planes that took off: "
    << num_takeoffs << endl
    << "Total number of planes left in landing queue: "
    << landing.size() << endl
    << "Total number of planes left in takeoff queue: "
    << takeoff.size() << endl
    << "Percentage of time runway idle: "
    << 100.0 * ((float) idle_time) / ((float) time) << "%" << endl
    << "Average wait in landing queue: "
    << ((float) land_wait) / ((float) num_landings) << " time units" << endl
    << "Average wait in takeoff queue: "
    << ((float) takeoff_wait) / ((float) num_takeoffs) << " time units" << endl
    << "Average observed rate of planes wanting to land: "
    << ((float) num_land_request) / ((float) time) << " per time unit" << endl
    << "Average observed rate of planes wanting to take off: "
    << ((float) num_takeoff_request) / ((float) time) << " per time unit" << endl;
}

#endif // RUNWAY_HPP