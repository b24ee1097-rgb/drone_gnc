// pid_controller in c
#include<math.h>
#include<stdio.h>
#include<string.h>
#include"pid_controller.h"
float constrain(float value ,const float minvalue, cont float maxvalue){
    return fminf(maxvalue,fmaxf(minvalue,value));
}
double pastAltitudeError, pastPitchError, pastRollError, pastYawRateError;
double pastVxError, pastVyError;
double altitudeIntegrator;

