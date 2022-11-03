#version 330 core
in vec2 texCoord;
out vec4 FragColor;
uniform sampler2D boundary_texture;	//boundary texture map specifying boundaries
uniform sampler2D state_texture3;	    //input texture containing f0, rho, ux and uy
uniform vec2 mousePos;

void main()
{
	//	TO DO: More sophisticated display output

  	vec2 pos = texCoord.xy;		//	Position of each lattice node	
	//	Following are for dummy display
	float color = texture2D( state_texture3, pos ).y;

	vec4 FragColorFluid = vec4( color*0.4, color*0.6, color, 0.0 );
	vec4 FragColorMask = texture(boundary_texture, texCoord);

	FragColor = FragColorFluid;

	//for (int i = 0; i < 3; i++) {
	//	 FragColor[i] = min(FragColorFluid[i], FragColorMask[i]);
	//}

	// float mask_color = texture2D(boundary_texture, pos).x;
    // if (mask_color > 0.0){
    //     FragColor =  FragColorFluid;
    // }else {
    //     FragColor =  FragColorMask;
    // }
}