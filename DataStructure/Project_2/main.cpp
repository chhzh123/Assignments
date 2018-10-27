//
//  main.cpp
//  AirportSim
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#include <iostream>
#include <queue>
#include <vector>
#include <cstring>
#include <algorithm>
#include <time.h>
#include <cmath>
#include "Random.hpp"
#include "Runway.hpp"
using namespace std;

int problem_num;
bool manual_mode = false;

void initialize(int& end_time,int& queue_limit,double& arrival_rate,double& departure_rate)
{
    switch (problem_num)
    {
        case 1:
        cout << "This program simulates Problem 1 - an airport with only one runway(#0)." << endl;
        break;
        case 2:
        cout << "This program simulates Problem 2 - an airport with two runways." << endl
             << "One only for landings(#0) and the other only for takeoffs(#1)." << endl;
        break;
        case 3:
        cout << "This program simulates Problem 3 - an airport with two runways." << endl
             << "One mainly for landings(#0) and the other mainly for takeoffs(#1)." << endl;
        break;
        case 4:
        cout << "This program simulates Problem 4 - an airport with three runways." << endl
             << "One only for landings(#0), the second one only for takeoffs(#1), and the third mainly for landings(#2)." << endl;
        break;
        case 5:
        cout << "This program simulates Problem 5 - an airport with only one runway(#0) and the planes have fuel level." << endl;
        break;
        default:
        cerr << "Error: No such problem!" << endl;
        return;
    }
    cout << "One plane can land or depart in each unit of time." << endl;
    cout << "Up to what number of planes can be waiting to land or take off at any time? " << flush;
    cin  >> queue_limit;
    cout << "How many units of time will the simulation run? " << flush;
    cin  >> end_time;
    if (manual_mode)
        return;
    bool acceptable = false;
    do {
        cout << "Expected number of arrivals per unit time? " << flush;
        cin >> arrival_rate;
        cout << "Expected number of departures per unit time? " << flush;
        cin >> departure_rate;
        if (arrival_rate < 0.0 || departure_rate < 0.0)
            cerr << "These rates must be nonnegative." << endl;
        else
            acceptable = true;
        if (acceptable && arrival_rate + departure_rate > 1.0)
            cerr << "Safety Warning: This airport will become saturated." << endl;
    } while (!acceptable);
}

int main(int argc, const char * argv[]) {
    problem_num = stoi(string(argv[1]));
    if (argc > 2) // the second argv is for Problem 6 (used for debug)
        manual_mode = (stoi(string(argv[2])) == 1 ? 1 : 0);
    int end_time, queue_limit, flight_number=0;
    double arrival_rate, departure_rate;
    initialize(end_time, queue_limit, arrival_rate, departure_rate); // initialize the arrive and depart list
    Random variable; // set the number of planes to takeoff and land

    // initialize the airport and runways
    Runway small_airport(queue_limit);
    if (problem_num == 1 || problem_num == 5)
        small_airport.set_runways(1,
            vector<Runway_mode> {Runway_mode::mainly_landing});
    else if (problem_num == 2)
        small_airport.set_runways(2,
            vector<Runway_mode> {Runway_mode::only_landing,
                                 Runway_mode::only_takeoff});
    else if (problem_num == 3)
        small_airport.set_runways(2,
            vector<Runway_mode> {Runway_mode::mainly_landing,
                                 Runway_mode::mainly_takeoff});
    else if (problem_num == 4)
        small_airport.set_runways(3,
            vector<Runway_mode> {Runway_mode::only_landing,
                                 Runway_mode::only_takeoff,
                                 Runway_mode::mainly_landing});

    // processing
    bool flag_change_status = false;
    for (int current_time = 0; current_time < end_time; current_time++) {
        // generate arrival and departure planes
        int num_arrivals;
        if (!manual_mode)
            num_arrivals = variable.poisson(arrival_rate);
        else
        {
            cout << "Please input the number of arrival planes at time " << current_time << ": ";
            cin >> num_arrivals;
        }

        if (problem_num != 5)
            for (int i = 0; i < num_arrivals; i++) {
                Plane current_plane(flight_number++, current_time, Status::arriving);
                if (small_airport.can_land(current_plane) != true) {
                    current_plane.refuse();
                }
            }
        else {
            vector<Plane> arrival_plane;
            for (int i = 0; i < num_arrivals; i++) {
                Plane current_plane(flight_number++, current_time, Status::arriving, end_time); // end_time-current_time
                arrival_plane.push_back(current_plane);
            }
            // sort by planes' fuel
            sort(arrival_plane.begin(),arrival_plane.end(),[](const Plane& plane1, const Plane& plane2)
                {return plane1.get_fuel() < plane2.get_fuel();});
            for (int i = 0; i < num_arrivals; i++) {
                if (small_airport.can_land(arrival_plane[i], (problem_num == 5 ? 1 : 0)) != true) {
                    arrival_plane[i].refuse();
                }
            }
        }
        // clear landing backlog
        if (problem_num == 3 && small_airport.landing_full() && !flag_change_status)
        {
            cout << "*** WARNING *** Too many planes arrive! Clear landing backlog..." << endl;
            small_airport.change_runway_status(0,Runway_mode::only_landing);
            small_airport.change_runway_status(1,Runway_mode::only_landing);
            flag_change_status = true;
        }

        int num_departures;
        if (!manual_mode)
            num_departures = variable.poisson(departure_rate);
        else
        {
            cout << "Please input the number of departure planes at time " << current_time << ": ";
            cin >> num_departures;
        }
        for (int j = 0; j < num_departures; j++) {
            Plane current_plane(flight_number++, current_time, Status::departing);
            if (small_airport.can_depart(current_plane) != true) {
                current_plane.refuse();
            }
        }

        // process the planes in the queue
        small_airport.activity(current_time, (problem_num == 5 ? 1 : 0));

        // Problem 3 - clear the backlog
        if  (problem_num == 3 && small_airport.landing_empty() && flag_change_status)
        {
            cout << "*** WARNING *** Landing backlog cleared." << endl;
            small_airport.change_runway_status(0,Runway_mode::mainly_landing);
            small_airport.change_runway_status(1,Runway_mode::mainly_takeoff);
            flag_change_status = false;
        }
    }
    small_airport.shut_down(end_time);
    return 0;
}