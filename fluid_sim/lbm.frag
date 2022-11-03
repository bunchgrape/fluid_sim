#version 330 core
in vec2 texCoord;
out vec4 FragColor[3];
uniform sampler2D boundary_texture;   //boundary texture map specifying boundaries
uniform sampler2D state_texture1;        //input texture containing f1-f4
uniform sampler2D state_texture2;        //input texture containing f5-f8
uniform sampler2D state_texture3;        //input texture containing f0, rho, ux and uy
uniform vec2 image_size;
uniform float tau;			//	Tau is corresponding to Viscosity and is used to evaluate feq (collision term).

void main()
{
    vec2 e[9];	//	9 lattice velocities
    float w[9];	//	9 lattice constants
    
	e[0] = vec2( 0, 0);
	e[1] = vec2( 1, 0);
	e[2] = vec2( 0, 1);
	e[3] = vec2(-1, 0);
	e[4] = vec2( 0,-1);
	e[5] = vec2( 1, 1);
	e[6] = vec2(-1, 1);
	e[7] = vec2(-1,-1);
	e[8] = vec2( 1,-1);

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

	float winWidth = image_size.x;
	float winHeight = image_size.y;

	vec2 pos = texCoord.xy;		//position of each lattice node	
	
	if ( texture( boundary_texture,pos ).x > 0.5 )
    {	//	Node is 'Fluid'
        float ff[9];// = {0.0};
        vec2 u = vec2(0.0,0.0);

		// streaming
		float f_star[9];
		f_star[0] = texture(state_texture3, pos)[0];
		for (int i = 1; i < 9; i++) {
			int block = (i - 1) / 4;
			int offset = (i - 1) % 4;

			vec2 pos_stream = pos - e[i] / image_size; // FIXME:
			float x = pos_stream.x;
			float y = pos_stream.y;

			if (block == 0) {
				f_star[i] = texture(state_texture1, pos_stream)[offset];
			} else {
				f_star[i] = texture(state_texture2, pos_stream)[offset];
			}
		}

		// density
		float rho = 0.0;
		for (int i = 0; i < 9; i++) {
			rho += f_star[i];
		}

		// velocity
		for (int i = 0; i < 9; i++) {
			u += e[i] * f_star[i];
		}
		u /= rho;

		// update f equilibrium
		float uu_dot = dot(u, u);
		float f_eq[9];
		for (int i = 0; i < 9; i++)
		{
			float eu_dot = dot(e[i], u);
			f_eq[i] = w[i] * rho * (1.0f + 3.0f * eu_dot + 4.5f * eu_dot * eu_dot - 1.5f * uu_dot);
		}

		for (int i = 0; i < 9; i++)
		{
			ff[i] = f_star[i] - (f_star[i] - f_eq[i]) / tau;
		}

      	FragColor[0] = vec4( ff[1], ff[2], ff[3], ff[4] );
      	FragColor[1] = vec4( ff[5], ff[6], ff[7], ff[8] );
      	FragColor[2] = vec4( ff[0], rho, u.x, u.y );     
	} 
    else 
    {	//	Node is 'Solid'
		//	To do: Handle the boundary condition here
		//	....

		float ff[9];// = {0.0};
        vec2 u = vec2(0.0,0.0);

		// streaming
		float f0[9];
		f0[0] = texture(state_texture3, pos)[0];
		for (int i = 1; i < 9; i++) {
			int block = (i - 1) / 4;
			int offset = (i - 1) % 4;

			vec2 pos_stream = pos - e[i] / image_size;
			float x = pos_stream.x * image_size.x;
			float y = pos_stream.y * image_size.y;

			if ((x < 2) || (x > (winWidth - 3)) || (y < 2) || (y > (winHeight - 3))) // boundaries
			{
				f0[i] = 0;
			} 
			else {
				if (block == 0) {
					f0[i] = texture(state_texture1, pos_stream)[offset];
				} else {
					f0[i] = texture(state_texture2, pos_stream)[offset];
				}
			}
		}
		
		// bounce back
		for (int i = 0; i < 9; i++) {
			int r = reverse[i];
			ff[i] = f0[r];
		}

		// density
		float rho = 0.0;

      	FragColor[0] = vec4( ff[1], ff[2], ff[3], ff[4] );
      	FragColor[1] = vec4( ff[5], ff[6], ff[7], ff[8] );
      	FragColor[2] = vec4( ff[0], rho, u.x, u.y );
	}

}