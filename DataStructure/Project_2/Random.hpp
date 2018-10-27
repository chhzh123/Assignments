//
//  Random.hpp
//  AirportSim
//
//  Copyright © 2018 曾天宇、陈鸿峥、黄杨峻. All rights reserved.
//

#ifndef RANDOM_HPP
#define RANDOM_HPP

class Random{
public:
    Random(bool pseudo=true){
        if(pseudo) seed = 1;
        else seed = time(NULL) % 0xFFFFFF;
        multiplier = 2743;
        add_on = 5923;
    }
    
    double rand_real(){
        double max = 0xFFFFFF;
        double temp = reseed();
        if (temp<0) {
            temp+=max;
        }
        return temp/max;
    }
    
    int poisson(double rate){
        double limit = exp(-rate);
        double prod = rand_real();
        int count = 0;
        while (prod>limit) {
            count++;
            prod *= rand_real();
        }
        return count;
    }
    
private:
    int reseed(){
        seed = seed*multiplier + add_on;
        return seed;
    }
    int seed,multiplier,add_on;
};

#endif // RANDOM_HPP