#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double pi = 3.1415926535;
#include <device_functions.h>

__device__ double factorial(double x)
{
    return  x <= 1 ? 1 : x * factorial(x - 1);
}

__device__ double getCn(int n, double x)
{
    double absx, sqrtx, nf, cn;
    absx = x;

    sqrtx = sqrt(absx);

    if (absx < 1.e-8)
    {
        nf = factorial(n);
        cn = (1.0 + x / ((n + 1.0) * (n + 2.0)) * (-1.0 + x / ((n + 3.0) * (n + 4.0)))) / nf;
    }
    else
    {
        if (x > 0)
        {
            if (n == 0)
            {
                cn = cos(sqrtx);
            }
            else if (n == 1)
            {
                cn = sin(sqrtx) / sqrtx;
            }
            else if (n == 2)
            {
                cn = (1 - cos(sqrtx)) / x;
            }
            else if (n == 3)
            {
                cn = (sqrtx - sin(sqrtx)) / (x * sqrtx);
            }
            else if (n > 3)
            {
                cn = -(getCn(n - 2, x) * 1 / factorial(n - 2)) / x;
            }
        }
        else
        {

            if (n == 0)
            {
                cn = cosh(sqrtx);
            }
            else if (n == 1)
            {
                cn = sinh(sqrtx) / sqrtx;
            }
            else if (n == 2)
            {
                cn = (cosh(sqrtx) - 1) / absx;
            }
            else if (n == 3)
            {
                cn = (sinh(sqrtx) - sqrtx) / (absx * sqrtx);
            }
            else if (n > 3)
            {
                cn = -(getCn(n - 2, x) * 1 / factorial(n - 2)) / x;
            }
        }
    }

    return cn;
}

__global__ void cnGpu(float x, float *cn)
{
    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

    int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    int tid = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;

    float x0 = getCn(tid, x);

    //__syncthreads();

    cn[tid] = x0;
}


void getCns(double x0, float *cn)
{
    float x = x0;
    float *dev_cn;

    cudaMalloc((void**)&dev_cn, 7*sizeof(float));
    cudaMemcpy(dev_cn, cn, 7*sizeof(float), cudaMemcpyHostToDevice);

    cnGpu << <1, 7 >> > (x, dev_cn);

    cudaMemcpy(cn, dev_cn, 7 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_cn);
}

double findX(int typeX, double ro, double SIG, double revs, double um)
{
    double x0 = 0.0, eps;

    if (typeX == 1)
    {
        double cbrt_1 = cbrt(pi * revs / (sqrtf(2) * (SIG - ro * um)));
        eps = cbrt_1 * um;
        x0 = 4.0 * pow((pi * revs + eps), 2);
    }
    else if (typeX == 2)
    {
        eps = pow((pi * (revs + 1) / ((2.0 / 3) * (pow(um, 3)) + SIG - ro * um)), (1.0 / 3.0)) * um;
        x0 = 4.0 * pow((pi * (revs + 1) - eps), 2);
    }
    else
    {
        printf("Unknown typeX, enter 1 or 2");
    }

    return x0;
}

double getX(int revs, double ro, double SIG, int typeX)
{
    double absRo, um, eps;

    absRo = fabs(ro);
    um = sqrt(1.0 - sqrt(2.0) * absRo);

    if (revs > 0)
    {
        return findX(typeX, ro, SIG, revs, um);
    }

    eps = pow((pi / (2.0 / 3.0 * pow(um, 3) + SIG - ro * um)), (1.0 / 3.0)) * um;

    return 4.0 * pow((pi - eps), 2);
}

double U(double x, double ro)
{
    float cn1x, cn2x, cn[7];

    getCns(x, cn);

    cn1x = cn[1];
    cn2x = cn[2];

    return sqrt(1.0 - ro * cn1x / sqrt(cn2x));
}

double dX(double x)
{
    return std::abs(0.0 - x);
}

double f(double x, double ro, const float* cn, double u)
{
    float cn2x, cn3x;

    cn2x = cn[2];
    cn3x = cn[3];

    return (cn3x / pow(sqrt(cn2x), 3)) * pow(u, 3) + ro * u;
}

double fx(double ro, const float* cn, double u)
{
    double c2, c3, c5, c6;

    c2 = cn[2];
    c3 = cn[3];
    c5 = cn[5];
    c6 = cn[6];

    return ((pow(c3, 2) - c5 + 4 * c6) / (4 * pow(sqrt(c2), 3)))
           * pow(u, 3) + (3 * (c3 / pow(sqrt(c2), 3)) * pow(u, 2) + ro)
                         * ((ro * sqrt(c2)) / 8 * u);
}

double newton(double x0, double ro, double sig)
{
    double prevX, u;
    float dev_x, * dev_cn, cn[7];

    u = U(x0, ro);
    getCns(x0, cn);

    prevX = dX(f(x0, ro, cn, u) - sig);

    while (prevX > 1.e-11)
    {
        u = U(x0, ro);

        x0 = x0 - ((f(x0, ro, cn, u) - sig) / fx(ro, cn, u));
        prevX = dX(f(x0, ro, cn, u) - sig);

        getCns(x0, cn);
    }

    return x0;
}

void lambert(const float* r0, const float* rk, float t, int revs, int typeX)
{
    double R1[3], R2[3], R1R2[3];
    float cn[7];
    double r1 = 0.0, r2 = 0.0, r1r2 = 0.0;
    double ro, x0;

    if (t < 0)
    {
        printf("Time is invalid");
        return;
    }

    for (int i = 0; i < 3; ++i)
    {
        R1[i] = r0[i];
        R2[i] = rk[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        r1 += pow(R1[i], 2);
        r2 += pow(R2[i], 2);
    }

    r1 = sqrt(r1);
    r2 = sqrt(r2);

    for (int i = 0; i < 3; ++i)
    {
        r1r2 += R1[i] * R2[i];
    }

    R1R2[0] = R1[1] * R2[2] - R1[2] * R2[1];
    R1R2[1] = R1[2] * R2[0] - R1[0] * R2[2];
    R1R2[2] = R1[0] * R2[1] - R1[1] * R2[0];

    double fi = acos(r1r2 / (r1 * r2));

    if (R1R2[2] < 0.0)
    {
        fi = 2 * pi - fi;
    }

    ro = sqrt(2.0 * r1 * r2) / (r1 + r2) * cos(fi / 2.0);

    double SIG = t / pow(sqrt(r1 + r2), 3);
    double SIGpar = 1.0 / 3.0 * (sqrt(2.0) + ro) * sqrt(1.0 - sqrt(2.0) * ro);

    if (SIG < SIGpar)
    {
        x0 = 0.0;
    }
    else
    {
        x0 = getX(revs, ro, SIG, typeX);
    }

    const double xSol = newton(x0, ro, SIG);

    getCns(x0, cn);

    double cn1x = cn[1];
    double cn2x = cn[2];
    double cn3x = cn[3];
    double uxro = U(xSol, ro);

    double s = sqrt((r1 + r2) / cn2x) * uxro;
    double f = 1 - pow(s, 2) * cn2x / r1;
    double g = t - pow(s, 3) * cn3x;

    double df = -s * cn1x / (r1 * r2);
    double dg = 1.0 - pow(s, 2) * cn2x / r2;

    double v1[3], v2[3];

    for (int i = 0; i < 3; ++i)
    {
        v1[i] = 1. / g * (R2[i] - f * R1[i]);
        v2[i] = df * R1[i] + dg * v1[i];
        printf("v1[%i] = %f, v2[%i] = %f \n", i, v1[i], i, v2[i]);
    }

    return;
}

int main() {
    double AU = 1.49597870691e8;
    double fMSun = 1.32712440018e11;             // km^3/sec^2

    double UnitR = AU;
    double UnitV = sqrt(fMSun / UnitR);          // km/sec
    double UnitT = (UnitR / UnitV) / 86400;         // day

    float unitT = 100.0 / UnitT;
    int typeX = 1.0, revs = 0.0;
    float r1[3] = { -7.8941608095246896e-01, -6.2501194900473045e-01, 3.5441335698377735e-05 };
    float r2[3] = { 1.3897892184188783e+00, 1.3377137029002054e-01, -3.1287386211010106e-02 };

    lambert(r1, r2, unitT, revs, typeX);
    //    return 0;
}