

__kernel void lbm(__global float* lbmData,
					__global float* boundaryData, 
					__global float4* textureData,
					unsigned int winWidth, unsigned int winHeight, float tau) {

	unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
	int index = y * winWidth + x;


	float e[9][2];	//	9 lattice velocities
    float w[9];	//	9 lattice constants
	e[0] = { 0, 0};
	e[1] = { 1, 0};
	e[2] = { 0, 1};
	e[3] = {-1, 0};
	e[4] = { 0,-1};
	e[5] = { 1, 1};
	e[6] = {-1, 1};
	e[7] = {-1,-1};
	e[8] = { 1,-1};

	w[0] = 4.0/9.0;
	w[1] = 1.0/9.0;
	w[2] = 1.0/9.0;
	w[3] = 1.0/9.0;
	w[4] = 1.0/9.0;
	w[5] = 1.0/36.0;
	w[6] = 1.0/36.0;
	w[7] = 1.0/36.0;
	w[8] = 1.0/36.0;

	

	textureData[y * winWidth + x] = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
}