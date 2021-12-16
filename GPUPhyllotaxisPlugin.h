#ifndef GPUPHYLLOTAXISPLUGIN_H
#define GPUPHYLLOTAXISPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum Boolean {
	TRUE = 1,
	FALSE = 0
};


typedef struct {
	float angle;
	float radius;
	float xCoordinate;
	float yCoordinate;
} Floret;

class GPUPhyllotaxisPlugin : public Plugin {
   public:
	   void input(std::string file);
	   void run();
	   void output(std::string file);
   private:
float calculateFloretRadius(const long long);
float calculateFloretAngle(const long long);
float calculateXCoordinate(const float, const float);
float calculateYCoordinate(const float, const float);
void printToFile(Floret *, const unsigned long long);
   private:
std::string inputfile;
std::string outputfile;
int OUTPUT_POSITION;
float SPIRAL_ANGLE;
int SPIRAL_POSITION;
float SPIRAL_SCALE;
int FLORETS_POSITION;
long long totalFlorets;
int showSpiral;
std::map<std::string, std::string> parameters;
Floret* florets;
Floret* resultFlorets;
};
__global__ void gpu_operations(Floret * florets, int SPIRAL_ANGLE, int SPIRAL_SCALE) {
   	long floretIndex = blockIdx.x*blockDim.x + threadIdx.x;
   	const float radius = SPIRAL_SCALE * sqrtf(floretIndex);
	const float angle = floretIndex * SPIRAL_ANGLE;
	const float x = radius * cosf(angle);
	const float y = radius * sinf(angle);
	florets[floretIndex].radius = radius;
	florets[floretIndex].angle = angle;
	florets[floretIndex].xCoordinate = x;
	florets[floretIndex].yCoordinate = y;
}


#endif
