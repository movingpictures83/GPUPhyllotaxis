#include "GPUPhyllotaxisPlugin.h"
float GPUPhyllotaxisPlugin::calculateFloretRadius(const long long floretNumber) {
	float radius = SPIRAL_SCALE * sqrt(floretNumber);
	return radius;
}

float GPUPhyllotaxisPlugin::calculateFloretAngle(const long long floretNumber) {
	float angle = floretNumber * SPIRAL_ANGLE;
	return angle;
}

float GPUPhyllotaxisPlugin::calculateXCoordinate(const float angle, const float radius) {
	const float x = radius * cosf(angle);
	return x;
}

float GPUPhyllotaxisPlugin::calculateYCoordinate(const float angle, const float radius) {
	const float y = radius * sinf(angle);
	return y;
}

void GPUPhyllotaxisPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 totalFlorets = atoll(parameters["totalFlorets"].c_str());
 showSpiral = atoi(parameters["showSpiral"].c_str());
 OUTPUT_POSITION = atoi(parameters["outputPosition"].c_str());
 SPIRAL_ANGLE = atof(parameters["spiralAngle"].c_str());
 SPIRAL_POSITION = atoi(parameters["spiralPosition"].c_str());
 SPIRAL_SCALE = atof(parameters["spiralScale"].c_str());
 FLORETS_POSITION = atoi(parameters["floretsPosition"].c_str());
}

void GPUPhyllotaxisPlugin::run() {
	printf("\n***Computing Florets in Parallel***\n");
	int numThreads = 1024;
  	long numCores = totalFlorets / 1024 + 1;
	printf("**Initializing Array**\n");
  	cudaMalloc(&florets, totalFlorets * sizeof(Floret));
	printf("**Executing Operations**\n");
	gpu_operations<<<numCores, numThreads>>>(florets, SPIRAL_ANGLE, SPIRAL_SCALE);
	resultFlorets = (Floret*) malloc(totalFlorets * sizeof(Floret));
	cudaMemcpy(resultFlorets, florets, totalFlorets * sizeof(Floret), cudaMemcpyDeviceToHost);
		long long floretIndex = 0;
		for (floretIndex = 0; floretIndex < totalFlorets; floretIndex++) {
			const float radius = resultFlorets[floretIndex].radius;
			const float angle = resultFlorets[floretIndex].angle;
			const float x = resultFlorets[floretIndex].xCoordinate;
			const float y = resultFlorets[floretIndex].yCoordinate;
			printf("Floret %lld, radius = % 5.2f, angle = % 5.2f, "
				"X coordinate = % 5.2f, Y coordinate = % 5.2f\n",
				floretIndex, radius, angle, x, y);
		}
	printf("***End Parallel Calculations***\n");
	cudaFree(&florets);

}

void GPUPhyllotaxisPlugin::output(std::string file) {
	FILE * outputFile = fopen(file.c_str(), "w");
	if (outputFile == NULL) {
		printf("*** Access to output file was denied ***");
	} else {
		long long floretIndex = 0;
		for (floretIndex = 0; floretIndex < totalFlorets; floretIndex++) {
			const float x = resultFlorets[floretIndex].xCoordinate;
			const float y = resultFlorets[floretIndex].yCoordinate;
			fprintf(outputFile, "%f %f\n", x, y);
		}
	}
	fclose(outputFile);
	free(resultFlorets);
}


PluginProxy<GPUPhyllotaxisPlugin> GPUPhyllotaxisPluginProxy = PluginProxy<GPUPhyllotaxisPlugin>("GPUPhyllotaxis", PluginManager::getInstance());

