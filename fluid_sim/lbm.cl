__kernel  void lbm_kernel(__global float* state_texture,
						__global float* state_texture_copy,
						__global float* boundary_texture, 
						__global float4* textureData,
						int winWidth, int winHeight, float tau) {

	int x = get_global_id(0);
    int y = get_global_id(1);
	int index = y * winWidth + x;

	float e[9][2];	//	9 lattice velocities
    float w[9];	//	9 lattice constants
	e[0][0] = 0; 
	e[0][1] = 0;
	e[1][0] = 1; 
	e[1][1] = 0;
	e[2][0] = 0; 
	e[2][1] = 1;
	e[3][0] =-1; 
	e[3][1] = 0;
	e[4][0] = 0; 
	e[4][1] =-1;
	e[5][0] = 1; 
	e[5][1] = 1;
	e[6][0] =-1; 
	e[6][1] = 1;
	e[7][0] =-1; 
	e[7][1] =-1;
	e[8][0] = 1; 
	e[8][1] =-1;


	w[0] = 4.0/9.0;
	w[1] = 1.0/9.0;
	w[2] = 1.0/9.0;
	w[3] = 1.0/9.0;
	w[4] = 1.0/9.0;
	w[5] = 1.0/36.0;
	w[6] = 1.0/36.0;
	w[7] = 1.0/36.0;
	w[8] = 1.0/36.0;

	int reverse[9];
	reverse[0] = 0;
	reverse[1] = 3;
	reverse[2] = 4;
	reverse[3] = 1;
	reverse[4] = 2;
	reverse[5] = 7;
	reverse[6] = 8;
	reverse[7] = 5;
	reverse[8] = 6;

	float ff[9]; // = {0.0};
	float ux = 0.0;
	float uy = 0.0;

	if (boundary_texture[3 * index] > 0.5) {
		// streaming
		float f_star[9];
		f_star[0] = state_texture[12 * index + 8];
		for (int i = 1; i < 9; i++) {
			int offset = (i - 1) % 12;

			int x_e = x - e[i][0];
			int y_e = y - e[i][0];
			int index_e = y_e * winWidth + x_e;
			f_star[i] = state_texture[12 * index_e + offset];
		}

		// density
		float rho = 0.0;
		for (int i = 0; i < 9; i++) {
			rho += f_star[i];
		}

		// velocity
		for (int i = 0; i < 9; i++) {
			ux += e[i][0] * f_star[i];
			uy += e[i][1] * f_star[i];
		}
		ux /= rho;
		uy /= rho;

		// update f equilibrium
		float uu_dot = (ux * ux + uy * uy);
		float f_eq[9];
		for (int i = 0; i < 9; i++)
		{
			float eu_dot = (e[i][0] * ux + e[i][1] * uy);
			f_eq[i] = w[i] * rho * (1.0f + 3.0f * eu_dot + 4.5f * eu_dot * eu_dot - 1.5f * uu_dot);
		}

		for (int i = 0; i < 9; i++)
		{
			ff[i] = f_star[i] - (f_star[i] - f_eq[i]) / tau;
		}


		for (int i = 0; i < 8; i++) {
			state_texture_copy[12 * index + i] = ff[i + 1];
		}
		state_texture_copy[12 * index + 8] = ff[0];
		state_texture_copy[12 * index + 9] = rho;
		state_texture_copy[12 * index + 10] = ux;
		state_texture_copy[12 * index + 11] = uy;

		textureData[index] = (float4)(ff[0], rho, ux, uy);
	}
	else {
		// streaming
		float f0[9];
		f0[0] = state_texture[12 * index + 8];
		for (int i = 1; i < 9; i++) {
			int offset = (i - 1) % 12;

			int x_e = x - e[i][0];
			int y_e = y - e[i][0];
			int index_e = y_e * winWidth + x_e;

			if ((x_e < 2) || (x_e > (winWidth - 3)) || (y_e < 2) || (y_e > (winHeight - 3))) // boundaries
			{
				f0[i] = 0;
			}
			else {
				f0[i] = state_texture[12 * index_e + offset];
			}
		}

		// bounce back
		for (int i = 0; i < 9; i++) {
			int r = reverse[i];
			ff[i] = f0[r];
		}

		// density
		float rho = 0.0;

		for (int i = 0; i < 8; i++) {
			state_texture_copy[12 * index + i] = ff[i + 1];
		}
		state_texture_copy[12 * index + 8] = ff[0];
		state_texture_copy[12 * index + 9] = rho;
		state_texture_copy[12 * index + 10] = ux;
		state_texture_copy[12 * index + 11] = uy;

		textureData[index] = (float4)(ff[0], rho, ux, uy);
	}
}